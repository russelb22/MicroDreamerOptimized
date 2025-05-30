@echo off
REM Batch file to run MicroDreamer with NVTX profiling using Nsight Systems

set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x86;%PATH%"

REM Set output directory
set OUTPUT_DIR=C:\3DMLGPU\p1\MicroDreamerOptimized\logdir\nsys\broad_nvtx

REM Create output directory if it doesn't exist
if not exist %OUTPUT_DIR% (
    mkdir %OUTPUT_DIR%
)

REM Profile MicroDreamer training loop only (skips NeRF fitting & video/image saving)
REM Run Nsight Systems with NVTX tracing and no sampling

nsys profile ^
  --trace=cuda,nvtx ^
  --sample=none ^
  --output="%OUTPUT_DIR%" ^
  --force-overwrite=true ^
  python main_nvtx.py ^
  --config=configs/image_sai.yaml ^
  --input test_data/05_objaverse_backpack_rgba.png ^
  --save_path 05_objaverse_backpack_rgba ^
  --profiling.enabled true ^
  --profiling.mode nvtx ^
  --profiling.scope broad ^
  --profiling.skip_postprocessing true ^
  --iters 20

REM python main_nvtx.py --config configs/image_sai.yaml input=test_data/05_objaverse_backpack_rgba.png save_path=05_objaverse_backpack_rgba --profiling.enabled=true --profiling.mode=nvtx --profiling.scope=broad