# MicroDreamerOptimized - 3D reconstruction accelerated with custom CUDA kernels
This repository was forked from the original MicroDreamer repository for use in our 3D Machine Learning with GPU Optimization course at the University of Washington, Seattle in the Spring of 2025. The aim of the project was to identify, profile, and optimize Python functions with custom CUDA kernel extensions.

Original work by ML-GSAI/MicroDreamer, used here under its Apache license.

## Project Overview and Results
These results are measured with NVTX Ranges and run in Google Colab

1. **gaussian_3d_coeff()**
   
   - Located in `gs_renderer.py`, this per-voxel function was highlighted by Nsight as an arithmetic hotspot with no external dependencies.  
   - Rewritten as a small CUDA kernel and launched via PyTorch’s C++ extension API.  
   - **Results:**  
     - CPU: ~1.3 ms per call  
     - GPU: ~0.12 ms per call  
     - **Speedup:** ≈ 11×  

3. **extract_fields()**
   
   -  Defined in gs_renderer.py, this method computes the occupancy value for every voxel in the 3D grid by calling gaussian_3d_coeff() inside a Python-level triple-nested loop.
   -  In our optimized version, we replace that entire loop (and the per-voxel Gaussian calls) with a single fused CUDA kernel. We launch one GPU thread per voxel in a 3D block/grid, and within each thread compute the Mahalanobis-weighted Gaussian sum directly—eliminating all Python loops and separate kernel invocations for a dramatic speedup.
     
   - **Tesla T4**  
     - CPU fallback: 9 164.7 ms  
     - Fused CUDA kernel: 42.7 ms  
     - **Speedup:** ≈ 215×  
   - **NVIDIA L4**  
     - CPU fallback: 8 199.5 ms  
     - Fused CUDA kernel: 24.3 ms  
     - **Speedup:** ≈ 338×
   - **NVIDIA A100**  
     - CPU fallback: 8 124.1 ms  
     - Fused CUDA kernel: 22.7 ms  
     - **Speedup:** ≈ 358×  

5. **End-to-end comparison**  
   - Baseline (all-CPU): 39.806 s  
   - CUDA `gaussian_3d_coeff` extension only: 37.521 s (≈ 1.06×)  
   - Fused CUDA `extract_fields` kernel (gauss + extract): 24.394 s (≈ 1.63×)  

## 📥 Installation -  Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/russelb22/MicroDreamerOptimized/blob/main/colab/MicroDreamerOptimized_Colab.ipynb)

## 📥 Installation -  Windows
### 1. Clone & Create Conda Environment
From a regular command prompt opened as Administrator:
```bash
# choose your installation folder (e.g. C:\mdo)
mkdir C:\mdo && cd C:\mdo

git clone https://github.com/russelb22/MicroDreamerOptimized 
cd MicroDreamerOptimized

conda env create -f environment.yml
conda activate mdo
```
### 2. Build MicroDreamerOptimized
```bash
install_md_opt.bat
```

### 3. Build CUDA extensions 
From an x64 Native command prompt as Admin  
``` bash
cd C:\mdo\MicroDreamerOptimized

conda activate mdo

install_extensions.bat  
```

At this point MDO is installed and CUDA extensions are built
The application should first be run without profiling since it takes a long time (8-10 mins) to initialize GPU caches on the first run:
```bash
run_main.bat  
```

### 4. Changes needed to run locally on a Windows machine with a GPU include: 

1. PATH has to include "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x86" or location of cl.exe

2. PATH has to include "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.1\target-windows-x64" or location of nsys.exe

3. pip install numpy==1.26.4 

4. conda install -c conda-forge ninja 

### NOTES:
1. To run with profiling use the -profile flag
2. Within run_main.bat, set USE_CUDA_GAUSS and USE_CUDA_EXTRACT to 0 or 1 depending on if you want the CUDA versions of the optimized functions to run.
3. When run with profiling, nsys will generate a .nsys-rep file in the MicroDreamerOptimized/logdir/nsys/*.nsys-rep.n This can be loaded in Nsight Systems.



## Usage Notes from Original Repository

Image-to-3D:

```bash
### preprocess
# background removal and recentering, save rgba at 256x256
python process.py test_data/name.jpg

# save at a larger resolution
python process.py test_data/name.jpg --size 512

# process all jpg images under a dir
python process.py test_data

### training gaussian stage
# train 20 iters and export ckpt & coarse_mesh to logs
python main_profile.py --config configs/image_sai.yaml input=test_data/name_rgba.png save_path=name_rgba

### training mesh stage
# auto load coarse_mesh and refine 3 iters, export fine_mesh to logs
python main2.py --config configs/image_sai.yaml input=test_data/name_rgba.png save_path=name_rgba
```

Image+Text-to-3D (ImageDream):

```bash
### training gaussian stage
python main.py --config configs/imagedream.yaml input=test_data/ghost_rgba.png prompt="a ghost eating hamburger" save_path=ghost_rgba
```

Calculate for CLIP similarity:
```bash
PYTHONPATH='.' python scripts/cal_sim.py
```

## More Results



https://github.com/user-attachments/assets/8888a353-df16-4e19-ac1b-7ee37ece7ed1




https://github.com/user-attachments/assets/7e52a87b-d1f6-4e7b-a6b4-7732ea69613c





## Acknowledgement

This work is built on many amazing open source projects, thanks to all the authors!

- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
- [LGM](https://github.com/3DTopia/LGM)
- [threestudio](https://github.com/threestudio-project/threestudio)


## BibTeX

```
@misc{chen2024microdreamerzeroshot3dgeneration,
      title={MicroDreamer: Zero-shot 3D Generation in $\sim$20 Seconds by Score-based Iterative Reconstruction}, 
      author={Luxi Chen and Zhengyi Wang and Zihan Zhou and Tingting Gao and Hang Su and Jun Zhu and Chongxuan Li},
      year={2024},
      eprint={2404.19525},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.19525}, 
}
```
