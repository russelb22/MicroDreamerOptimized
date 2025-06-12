# MicroDreamerOptimized
This repository was forked from the original MicroDreamer repository for use in our 3D Machine Learning with GPU Optimization course at the University of Seattle, Washington in the Spring of 2025. The aim of the project was to identify, profile, and optimize Python functions with CUDA extensions.

https://github.com/user-attachments/assets/0a99424a-2e7a-47f0-9f0a-b6713b7686b5

## Installation

From a regular command prompt, run the following command: conda activate C:\conda\pynerf2

2. mkdir C:\mdo
3. cd C:\mdo
4. git clone https://github.com/russelb22/MicroDreamerOptimized 
5. cd MicroDreamerOptimized  
6. run install_md_opt.bat batch file to do the install (takes about 8 minutes):
install_md_opt.bat

Steps to build CUDA extension:
1. Start an x64 Native command prompt as Admin  
2. Within x64 Native command prompt, execute ‘conda activate C:\conda\pynerf2' 
3. Run the following batch file to install CUDA extensions  
install_extensions.bat  

Run the application without profiling since it takes a long time (10-15 mins) to initialize GPU caches the first time the application is run with the following command 
run_main.bat  

NOTES:
1. The application can be run using the batch file run_main.bat
2. To run without profiling do not use the -profile flag
3. Within run_main.bat, set USE_CUDA_GAUSS and USE_CUDA_EXTRACT to 0 or 1 depending on if you want the CUDA versions of the optimized functions to run

Running with profiling
1. Once the application is run once then it can be run with profiling as follows:  
run_main.bat -profile  
2. This will generate a .nsys-rep file in the MicroDreamerOptimized/logdir/nsys/*.nsys-rep which can be loaded in Nsight Systems 

At this point MD is installed and CUDA extensions are built, so run_main.bat can be used with or without -profile and with USE_CUDA_GAUSS and USE_CUDA_EXTRACT each set to 0 or 1. 

## Usage

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
python main.py --config configs/image_sai.yaml input=test_data/name_rgba.png save_path=name_rgba

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
