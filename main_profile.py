import os
import cv2
import time
import tqdm
import numpy as np
from diffusers import DDIMScheduler
import torch
import torch.nn.functional as F
from torch.cuda import nvtx

import rembg
import math

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam, gaussian_3d_coeff

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
from torchvision.utils import save_image
from core.options import config_defaults
from convert import Converter
from gs_postprocess import filter_out
from loss_utils import ssim,lpips
# from clip_sim import cal_clip_sim

from torch.cuda import nvtx

# Global profiler object
global_profiler = {"broad": None, "function": None}

def is_nvtx(opt):
    return opt.profiling.enabled and opt.profiling.mode.lower() == "nvtx"

def is_torch(opt):
    return opt.profiling.enabled and opt.profiling.mode.lower() == "torch"

# --- NVTX helpers ---

def nvtx_push(opt, fn_name: str, scope: str):
    #print(f"[DEBUG] NVTX CHECK: enabled={opt.profiling.enabled}, mode={opt.profiling.mode}, scope={opt.profiling.scope}")
    if is_nvtx(opt) and opt.profiling.scope.lower() == scope:
        nvtx.range_push(fn_name)

def nvtx_pop(opt, scope: str):
    if is_nvtx(opt) and opt.profiling.scope.lower() == scope:
        nvtx.range_pop()

def nvtx_push_broad(opt, fn_name: str):
    #print(f"[DEBUG] NVTX PUSH BROAD: {fn_name}")
    nvtx_push(opt, fn_name, "broad")

def nvtx_pop_broad(opt): nvtx_pop(opt, "broad")
def nvtx_push_function(opt, fn_name: str): nvtx_push(opt, fn_name, "function")
def nvtx_pop_function(opt): nvtx_pop(opt, "function")

# --- Torch Profiler helpers ---

def torch_profiler_start(opt, scope: str):
    if is_torch(opt) and opt.profiling.scope.lower() == scope:
        global_profiler[scope] = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(opt.profiling.output_dir),
            record_shapes=True,
            with_stack=True,
        )
        global_profiler[scope].__enter__()
        global_profiler[scope].start()

def torch_profiler_step(opt, scope: str):
    if is_torch(opt) and opt.profiling.scope.lower() == scope:
        global_profiler[scope].step()

def torch_profiler_stop(opt, scope: str):
    if is_torch(opt) and opt.profiling.scope.lower() == scope:
        global_profiler[scope].stop()
        global_profiler[scope].__exit__(None, None, None)
        global_profiler[scope] = None

def torch_profiler_start_broad(opt): torch_profiler_start(opt, "broad")
def torch_profiler_step_broad(opt): torch_profiler_step(opt, "broad")
def torch_profiler_stop_broad(opt): torch_profiler_stop(opt, "broad")

def torch_profiler_start_function(opt): torch_profiler_start(opt, "function")
def torch_profiler_step_function(opt): torch_profiler_step(opt, "function")
def torch_profiler_stop_function(opt): torch_profiler_stop(opt, "function")

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = 0

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        self.init_3d=True   
        # self.scheduler=DDIMScheduler(clip_sample=False)
        import json
        self.config =json.load(open("./scheduler_config.json"))
        self.scheduler=DDIMScheduler.from_config(self.config)
        self.denoise_steps=self.opt.denoise_steps
        self.scheduler.set_timesteps(self.denoise_steps)
        self.total_steps=self.opt.total_steps-1
        self._denoise_step=0


    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed
    def get_denoise_schedule(self):
        
        num_train_timesteps = 1000
        start = self.denoise_steps*(1-self.opt.t_start)
        end = self.denoise_steps*(1-self.opt.t_end)
        # if self.cfg.timesche_type.endswith("linear"):
        dsp=int((self._denoise_step/self.total_steps)*(end-start)+start)

        t=self.scheduler.timesteps[dsp]
        return t
    def get_reconstruct_steps(self):
        steps_max=self.opt.steps_max
        steps_min=self.opt.steps_min
        reconstruct_steps_schedule= self.opt.steps_schedule

        if reconstruct_steps_schedule == 'cosine_up':
            reconstruct_steps = (-math.cos(self._denoise_step/self.total_steps*math.pi)+1)/2*(
                steps_max-steps_min)+steps_min
            
        elif reconstruct_steps_schedule == 'cosine_down':
            reconstruct_steps = (math.cos(self._denoise_step/self.total_steps*math.pi)+1)/2*(
                steps_max-steps_min)+steps_min
        
        elif reconstruct_steps_schedule == 'linear':
            reconstruct_steps = self._denoise_step/self.total_steps * \
                (steps_max -
                 steps_min)+steps_min
        elif reconstruct_steps_schedule == 'cosine_up_then_down':
            reconstruct_steps = (-math.cos(self._denoise_step/self.total_steps*2*math.pi)+1)/2*(
                steps_max-steps_min)+steps_min

        else:
            reconstruct_steps = steps_max

        return int(reconstruct_steps)

    def get_batch_size(self):
        batch_size_max=self.opt.batch_size_max
        batch_size_min=self.opt.batch_size_min
        batch_size = self._denoise_step/self.total_steps * \
                (batch_size_max -batch_size_min)+batch_size_min

        return int(batch_size)
    
    def get_ref_loss(self):
        ref_loss_max=self.opt.ref_loss
        ref_loss_min=0.01
        ref_loss = self._denoise_step/self.total_steps * \
                (ref_loss_max - ref_loss_min)+ref_loss_min

        return ref_loss

    def prepare_train(self):

        self.step = 0

        # setup training
        nvtx_push_broad(opt, "GridEncoder Setup")
        self.renderer.gaussians.training_setup(self.opt)
        nvtx_pop_broad(opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()


        batch_size=self.get_batch_size()

        nvtx_push_broad(opt, "TRAIN_LOOP")
        for _ in range(self.train_steps):
            nvtx_push_broad(opt, f"TRAIN_STEP {self._denoise_step}")
            target_img=None

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### novel view (manual batch)
            # render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            render_resolution = 256
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)


            cur_cams = []

            hor_base=np.random.randint(-180, -180+360//batch_size)
            
            bg_color = torch.tensor([1, 1, 1] , dtype=torch.float32, device="cuda")
            
            for i in range(batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                if self.opt.even_view:
                    hor = hor_base+(360//batch_size)*i
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                cur_cams.append(cur_cam)
                    
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            recon_steps=self.get_reconstruct_steps()
            if self.init_3d:
                recon_steps=self.opt.init_steps
            step_t=self.get_denoise_schedule()
            for _1 in range(recon_steps):
                final_step = (self._denoise_step == self.total_steps-1) and _1 == (recon_steps-1)
                self.step += 1
                step_ratio = min(1, self.step / self.opt.iters)

                # update lr
                self.renderer.gaussians.update_learning_rate(self.step)
                loss=0.0

                ### known view
                if self.input_img_torch is not None and not self.opt.imagedream:
                    cur_cam = self.fixed_cam
                    
                    # rendering views
                    nvtx_push_broad(opt, "RENDERING")
                    out = self.renderer.render(cur_cam)
                    nvtx_pop_broad(opt)

                    # rgb loss
                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                    loss = loss + self.opt.ref_loss * F.l1_loss(image, self.input_img_torch,reduction='sum')

                    # # mask loss
                    mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                    loss = loss + self.opt.ref_mask_loss * F.mse_loss(mask, self.input_mask_torch,reduction='sum')
                images=[]
                
                nvtx_push_broad(opt, "RENDERING NOVEL VIEWS")
                for i, cur_cam in enumerate(cur_cams):
                    nvtx_push_broad(opt, f"RENDER VIEW {i}")
                    out = self.renderer.render(cur_cam,bg_color=bg_color)
                    nvtx_pop_broad(opt)
                    image=out["image"].unsqueeze(0)
                    images.append(image)
                images=torch.cat(images,dim=0)
                nvtx_pop_broad(opt)


                if self.enable_sd:
                    if self.opt.mvdream or self.opt.imagedream:
                        nvtx_push_broad(opt, f"GUIDANCE SD step {self._denoise_step}")
                        target_img = self.guidance_sd.train_step(images, poses, step_ratio=None,guidance_scale=self.opt.cfg,target_img=target_img,step=step_t,init_3d=self.init_3d,iter_steps=self.denoise_steps)
                        nvtx_pop_broad(opt)
                        loss_my = F.l1_loss(images, target_img.to(images), reduction='sum')/images.shape[0]
                        loss = loss + self.opt.lambda_sd * loss_my

                if self.enable_zero123:
                    nvtx_push_broad(opt, f"GUIDANCE ZERO123 step {self._denoise_step}")
                    target_img=self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=None, default_elevation=self.opt.elevation,guidance_scale=self.opt.cfg,target_img=target_img,step=step_t,init_3d=self.init_3d,iter_steps=self.denoise_steps,inverse_ratio=self.opt.inv_r,ddim_eta=self.opt.eta)
                    nvtx_pop_broad(opt)

                    nvtx_push_broad(opt, f"Loss calculation {self._denoise_step}")
                    loss_my = F.l1_loss(images, target_img.to(images), reduction='sum')/images.shape[0]
                    nvtx_pop()

                    # + torch.prod(torch.tensor(images.shape[1:]))*(1-ssim(images,target_img.to(images)))
                    
                    loss = loss + self.opt.lambda_zero123 * loss_my
            
                # backward pass
                # print("calling nvtx_push_broad for BACKWARD PASS")
                nvtx_push_broad(opt, f"BACKWARD PASS step {self._denoise_step}")
                if torch.is_tensor(loss) and loss.requires_grad:
                    loss.backward()
                #else:
                #    print(f"[WARN] Skipped loss.backward(), loss is type {type(loss)} at step {self._denoise_step}")
                nvtx_pop_broad(opt)

                # optimize step
                nvtx_push_broad(opt, f"OPTIMIZER.STEP step {self._denoise_step}")
                self.optimizer.step()
                nvtx_pop_broad(opt)
                
                nvtx_push_broad(opt, f"OPTIMIZER.ZERO_GRAD step {self._denoise_step}")
                self.optimizer.zero_grad()
                nvtx_pop_broad(opt)

                # densify and prune
                if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                    viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                    self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    #self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if viewspace_point_tensor.grad is not None:
                        self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    #else:
                    #    print(f"[WARN] viewspace_point_tensor.grad is None at step {self.step}, skipping densification stats.")

                    if (self.step % self.opt.densification_interval == 0) or final_step:
                        self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.005, extent=4, max_screen_size=1)
                    
                    
            self._denoise_step += 1 if not self.init_3d else 0
            self.init_3d=False
            nvtx_pop_broad(opt)
        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True
        nvtx_pop_broad(opt)

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)


    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()


    @torch.no_grad()
    def save_video(self, path):
        import imageio
        # vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
        # hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]
        vers=[0]*120
        hors=list(range(0,180,3))+list(range(-180,0,3))

        # vers=vers[:8]
        # hors=hors[:8]

        render_resolution = 512

        import nvdiffrast.torch as dr

        if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            glctx = dr.RasterizeGLContext()
        else:
            glctx = dr.RasterizeCudaContext()

        rgbs_ls=[]
        for ver, hor in zip(vers, hors):
            # render image
            pose = orbit_camera(ver, hor, self.cam.radius)

            cur_cam = MiniCam(
                pose,
                render_resolution,
                render_resolution,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            
            cur_out = self.renderer.render(cur_cam)

            rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            rgbs_ls.append(rgbs)
        
        rgbs = torch.cat(rgbs_ls, dim=0).permute(0,2,3,1).cpu().numpy()
        rgbs= [rgbs[i] for i in range(rgbs.shape[0])]
        imageio.mimsave(path, rgbs, fps=30)
        
        # save_image(rgbs,path,padding=0)

    @torch.no_grad()
    def save_image(self, path,num=8):
        os.makedirs(path,exist_ok=True)
        vers=[0]*num
        hors = np.linspace(-180, 180, num, dtype=np.int32, endpoint=False).tolist()

        # vers=vers[:8]
        # hors=hors[:8]

        render_resolution = 512

        import nvdiffrast.torch as dr

        if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            glctx = dr.RasterizeGLContext()
        else:
            glctx = dr.RasterizeCudaContext()

        rgbs_ls=[]
        cnt=0
        for ver, hor in zip(vers, hors):
            # render image
            pose = orbit_camera(ver, hor, self.cam.radius)

            cur_cam = MiniCam(
                pose,
                render_resolution,
                render_resolution,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            
            cur_out = self.renderer.render(cur_cam)

            rgbs = cur_out["image"] # [3, H, W] in [0, 1]
            save_image(rgbs,os.path.join(path,f"{cnt}.png"),padding=0)
            cnt+=1


    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        # assert 0
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')

            print("calling extract_mesh from within mode == geo block in save_model")
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.' + self.opt.mesh_format)
            
            print("calling extract_mesh from within mode == geo+tex block in save_model")
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            print("returned from calling extract_mesh from within mode == geo+tex block in save_model")
            
            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")
        
        
    def save_mesh(self):
        path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
        opt_ = config_defaults['big']
        opt_.test_path=path
        opt_.force_cuda_rast = self.opt.force_cuda_rast
        converter = Converter(opt_).to(self.device)
        converter.fit_nerf()
        converter.fit_mesh()
        converter.fit_mesh_uv(padding=16)
        converter.export_mesh(path.replace('.ply', '.obj'))

    # no gui mode
    def train(self, iters=31):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        filter_out(self.renderer)
        
        # save
        #if not (opt.profiling.enabled and opt.profiling.get("skip_postprocessing", False)):
        print("[DEBUG] calling save_model which should call gaussian_3D_coeff, which has been put onto CUDA")
        #self.save_model(mode='model')
        self.save_model(mode='geo+tex')
        #self.save_mesh()
        
import os
import torch
import math

USE_CUDA_KERNEL = os.getenv("USE_CUDA_KERNEL", "0") == "1"

if USE_CUDA_KERNEL:
    import cuda_kernels

    def compare_gaussian_cpu_to_gpu():
        N = 100000
        torch.manual_seed(42)  # Ensures reproducibility

        # Generate identical input data for both CPU and GPU
        xyzs = torch.randn(N, 3, dtype=torch.float32)
        covs = torch.randn(N, 6, dtype=torch.float32)

        # CPU (reference) version: uses the pure-PyTorch fallback
        ref_result = gaussian_3d_coeff(xyzs, covs).cpu()

        # GPU (CUDA kernel) version
        xyzs_cuda = xyzs.to('cuda').contiguous()
        covs_cuda = covs.to('cuda').contiguous()
        out = torch.empty(N, device='cuda', dtype=torch.float32)
        cuda_kernels.gaussian_3d_launcher(xyzs_cuda, covs_cuda, out)
        out_cpu = out.cpu()

        # Check for closeness
        if torch.allclose(ref_result, out_cpu, rtol=1e-4, atol=1e-6):
            print("CUDA output matches reference implementation.")
        else:
            max_diff = torch.max(torch.abs(ref_result - out_cpu))
            print("Mismatch detected. Max difference:", max_diff.item())

        # Print sample values from both for visual inspection
        print("\nFirst 10 values from CPU version:")
        print(ref_result[:10].numpy())

        print("\nFirst 10 values from CUDA version:")
        print(out_cpu[:10].numpy())

    def compare_extract_fields_cpu_to_gpu():
        # Use a small resolution so the pure-PyTorch version is reasonably fast
        resolution = 16
        num_blocks = 4   # then split_size = resolution // num_blocks = 4
        relax_ratio = 1.5

        # Number of Gaussians
        N0 = 50
        torch.manual_seed(123)

        # Random Gaussian centers in [-1,1]
        means = (torch.rand(N0, 3, dtype=torch.float32) * 2.0) - 1.0

        # Build diagonal covariance -> invert to get inv_cov6
        variances = torch.rand(N0, 3, dtype=torch.float32) * 0.1 + 0.05
        inv_a = 1.0 / variances[:, 0]
        inv_b = torch.zeros(N0, dtype=torch.float32)
        inv_c = torch.zeros(N0, dtype=torch.float32)
        inv_d = 1.0 / variances[:, 1]
        inv_e = torch.zeros(N0, dtype=torch.float32)
        inv_f = 1.0 / variances[:, 2]
        inv_cov6 = torch.stack([inv_a, inv_b, inv_c, inv_d, inv_e, inv_f], dim=1)

        # Random opacities in [0,1]
        opacities = torch.rand(N0, dtype=torch.float32)

        # --------------------------------------------
        # CPU reference with EXACT same culling logic
        # --------------------------------------------
        block_size = 2.0 / num_blocks            # each sub-cube width in world coords
        span = relax_ratio * block_size          # culling threshold

        # Build a grid of voxel centers in [-1,1]
        #coords = torch.linspace(-1.0, 1.0, resolution)
        coords = -1.0 + (2 * torch.arange(resolution, dtype=torch.float32) + 1)/resolution
        xs, ys, zs = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], dim=1)  # [M,3]
        M = pts.shape[0]  # M = resolution^3

        occ_flat_cpu = torch.zeros(M, dtype=torch.float32)

        # Loop over voxels in pure Python/PyTorch, but with vectorized inner loop over Gaussians
        # We will do "for each voxel index m, for each i in [0..N0-1], check cull, then accumulate"
        for m in range(M):
            fx = pts[m, 0].item()
            fy = pts[m, 1].item()
            fz = pts[m, 2].item()

            total = 0.0
            for i in range(N0):
                mx = means[i, 0].item()
                my = means[i, 1].item()
                mz = means[i, 2].item()

                # Same culling as GPU kernel
                if abs(fx - mx) > span or abs(fy - my) > span or abs(fz - mz) > span:
                    continue

                dx = fx - mx
                dy = fy - my
                dz = fz - mz

                inv_a_cpu = inv_cov6[i, 0].item()
                inv_b_cpu = inv_cov6[i, 1].item()
                inv_c_cpu = inv_cov6[i, 2].item()
                inv_d_cpu = inv_cov6[i, 3].item()
                inv_e_cpu = inv_cov6[i, 4].item()
                inv_f_cpu = inv_cov6[i, 5].item()

                # Mahalanobis squared
                t0 = dx * inv_a_cpu + dy * inv_b_cpu + dz * inv_c_cpu
                t1 = dx * inv_b_cpu + dy * inv_d_cpu + dz * inv_e_cpu
                t2 = dx * inv_c_cpu + dy * inv_e_cpu + dz * inv_f_cpu
                sq = dx * t0 + dy * t1 + dz * t2

                power = -0.5 * sq
                if power > 0.0:
                    continue
                w = math.exp(power)

                total += w * float(opacities[i].item())

            occ_flat_cpu[m] = total

        occ_cpu = occ_flat_cpu.view(resolution, resolution, resolution)

        # --------------------------------------------
        # GPU version
        # --------------------------------------------
        means_cuda     = means.to('cuda')
        inv_cov6_cuda  = inv_cov6.to('cuda')
        opacities_cuda = opacities.to('cuda')

        occ_gpu = cuda_kernels.extract_fields_launcher(
            means_cuda.contiguous(),
            inv_cov6_cuda.contiguous(),
            opacities_cuda.contiguous(),
            resolution,
            num_blocks,
            float(relax_ratio)
        )
        occ_gpu_cpu = occ_gpu.cpu()

        # Compare
        if torch.allclose(occ_cpu, occ_gpu_cpu, rtol=1e-4, atol=1e-6):
            print("extract_fields GPU output matches CPU reference.")
        else:
            diff = torch.abs(occ_cpu - occ_gpu_cpu)
            max_diff = diff.max()
            print("Mismatch detected in extract_fields. Max difference:", max_diff.item())

        # Print a sample slice for visual inspection
        z_slice = resolution // 2
        print(f"\nCPU occ slice at z={z_slice}:")
        print(occ_cpu[:, :, z_slice])
        print(f"\nGPU occ slice at z={z_slice}:")
        print(occ_gpu_cpu[:, :, z_slice])


if __name__ == "__main__":
    import argparse
    import sys
    from omegaconf import OmegaConf

    #compare_extract_fields_cpu_to_gpu()
    #sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)
    gui.seed_everything()

    print("Profiling Mode:", opt.profiling.mode)
    print("[DEBUG] Profiling config after CLI merge:", opt.profiling)
    print("[DEBUG] Iters specified:", opt.iters)
    print("[DEBUG] Total_steps specified:", opt.total_steps)

    print("START TIMER")
    t0 = time.perf_counter()

    if opt.gui:
        nvtx_push_broad(opt, "RENDER_TOP_LEVEL")
        gui.render()
        nvtx_pop_broad(opt)
    else:
        #print("calling nvtx_push_broad for TRAIN_TOP_LEVEL - gui.train (only calling NVTX on scope broad tho")
        nvtx_push_broad(opt, "TRAIN_TOP_LEVEL")
        gui.train(opt.total_steps)
        nvtx_pop_broad(opt)
        
    # gui.save_video(f'./{opt.save_path}-video.mp4')

    #lets just focus on training for now by using the skip_postprocessing=true command line argument
    if not (opt.profiling.enabled and opt.profiling.get("skip_postprocessing", False)):
        #nvtx.range_push("save_image")
        print("calling gui.save_image")
        gui.save_image(f'./test_dirs/work_dirs/{opt.save_path}',num=8)
        #nvtx.range_pop()

        gui.save_video(f'./test_dirs/work_dirs/{opt.save_path}/video.mp4')
    
    t1 = time.perf_counter()
    print(f"END TIMER: Total elapsed time: {t1 - t0:.3f} seconds")
    
    # Save timing to summary log if RUN_LABEL is set
    label = os.environ.get("RUN_LABEL", "unnamed_run")
    summary_path = os.path.join("logdir", "timing_summary.txt")
    os.makedirs("logdir", exist_ok=True)
    with open(summary_path, "a") as f:
        f.write(f"{label}: {t1 - t0:.3f} seconds\n")
