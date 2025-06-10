@echo off
setlocal

set USE_CUDA_GAUSS=1
set USE_CUDA_EXTRACT=1

echo [INFO]USE_CUDA_EXTRACT=%USE_CUDA_EXTRACT%
echo [INFO]USE_CUDA_GAUSS=%USE_CUDA_GAUSS%

:: Launch Nsight Compute
ncu --set full ^
    --kernel-name extract_fields_kernel ^
    --launch-count 1 ^
    --target-processes all ^
    --export extract_fields_ncu_report.csv ^
    python main_profile.py ^
        --config=configs/image_sai.yaml ^
        --input=test_data/05_objaverse_backpack_rgba.png ^
        --save_path=05_objaverse_backpack_rgba ^
        --profiling.enabled=true ^
        --profiling.mode=nvtx ^
        --profiling.scope=function

pause