@echo off
REM Batch file to run MicroDreamer under Nsight Compute
REM (assumes you have already activated your conda environment)

REM 2) Launch Nsight Compute, profiling only the extract_fields_kernel launch
REM    --set full                     : capture the full default metrics set
REM    --kernel-name extract_fields_kernel : collect metrics only for that kernel
REM    --launch-count 1               : profile only the first invocation of that kernel
REM    --target-processes all         : include any subprocesses or threads
REM    --output extract_fields_ncu_report       : write extract_fields_ncu_report.ncu-rep
REM    --export extract_fields_ncu_report.csv   : also write CSV summary
REM    -- <python invocation>         : everything after "--" is your normal python command

ncu ^
    --set full ^
    --kernel-name extract_fields_kernel ^
    --launch-count 1 ^
    --target-processes all ^
    --output extract_fields_ncu_report ^
    --export extract_fields_ncu_report.csv ^
    -- ^
    python main_profile.py ^
         --config=configs/image_sai.yaml ^
         --input=test_data/05_objaverse_backpack_rgba.png ^
         --save_path=05_objaverse_backpack_rgba ^
         --profiling.enabled=true ^
         --profiling.mode=nvtx ^
         --profiling.scope=function ^
         --iters=20

REM 3) Pause so you can see any messages before the window closes
pause
