@echo off
REM Batch file to run MicroDreamer with or without NVTX profiling via Nsight Systems

REM Set base output directory
set BASE_OUTPUT_DIR=.\logdir\nsys

REM Generate a timestamp (format: YYYYMMDD_HHMMSS)
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do (
    set mm=%%a
    set dd=%%b
    set yyyy=%%c
)
for /f "tokens=1-2 delims=: " %%a in ("%time%") do (
    set hh=%%a
    set min=%%b
)
REM remove leading spaces or zero pad if needed
if 1%hh% LSS 110 set hh=0%hh%

set TIMESTAMP=%yyyy%%mm%%dd%_%hh%%min%

REM Final output path with timestamp
set OUTPUT_FILE=%BASE_OUTPUT_DIR%\profile_%TIMESTAMP%

REM Add cl.exe location to PATH
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x86;%PATH%"

REM Create output directory if it doesn't exist
if not exist %BASE_OUTPUT_DIR% (
    mkdir %BASE_OUTPUT_DIR%
)

REM Enable (1) or Disable (0) CUDA kernel
set USE_CUDA_KERNEL=0

echo [INFO]USE_CUDA_KERNEL=%USE_CUDA_KERNEL%

REM Check if the first argument is -profile
IF "%1" == "-profile" (
    echo [INFO] Running with Nsight Systems NVTX profiling...
    echo [INFO] Output file: %OUTPUT_FILE%.nsys-rep
    
    nsys profile ^
        --trace=cuda,nvtx ^
        --sample=none ^
        --output="%OUTPUT_FILE%" ^
        --force-overwrite=true ^
        python main_profile.py ^
        --config=configs/image_sai.yaml ^
        --input test_data/05_objaverse_backpack_rgba.png ^
        --save_path 05_objaverse_backpack_rgba ^
        --profiling.enabled=true ^
        --profiling.mode=nvtx ^
        --profiling.scope=function ^
        --iters=20
) ELSE (
    echo [INFO] Running without profiling...

    python main_profile.py ^
        --config=configs/image_sai.yaml ^
        --input test_data/05_objaverse_backpack_rgba.png ^
        --save_path 05_objaverse_backpack_rgba ^
        --profiling.enabled false ^
        --iters 500
)
