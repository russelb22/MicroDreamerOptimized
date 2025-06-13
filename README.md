# MicroDreamerOptimized
This repository was forked from the original MicroDreamer repository for use in our 3D Machine Learning with GPU Optimization course at the University of Seattle, Washington in the Spring of 2025. The aim of the project was to identify, profile, and optimize Python functions with CUDA extensions.

Original work by ML-GSAI/MicroDreamer, used here under its Apache license.

## Project Overview and Results - CUDA Optimization Summary
In this project we used NVIDIA Nsight Systems to profile MicroDreamer and identify pure-Python hotspots worth moving into custom CUDA kernels.

**1.	gaussian_3d_coeff()**

  •	Located in gs_renderer.py, this per-voxel function was highlighted by Nsight as a heavy arithmetic hotspot with no external dependencies.

  •	We rewrote it as a small CUDA kernel and launched it via PyTorch’s C++ extension API.

  •	Result: CPU: 252 ms → GPU: 0.112 ms → **≈ 2.24×10¹ speedup.**

 
**2.	extract_fields()**

  •	Originally, this triple-nested loop called gaussian_3d_coeff() for every voxel in the 3D grid, costing **17.367 s.**

  •	We fused the loop and the Gaussian math into one 3D CUDA kernel (one thread per voxel).
  
  •	**On the T4 GPU**: 17.367 s → 0.1817 s → **≈ 95.6× speedup.**

 
**3.	Effect of a more powerful GPU**

  •	Moving from the AWS T4 to an AWS A10G further accelerated our fused kernel to **21.36 ms.**

  •	**Relative to the original Python version:** 17.367 s → 0.02136 s → **≈ 813× overall speedup.**

 
**4.	End-to-end comparison**

  •	If we compare the raw Python extract_fields() time (17.367 s) to the kernel-only execution on the A10G (10.526 ms, measured with NVTX), we see a remarkable ≈ **1 650× overall speedup.**
  
## Installation
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
