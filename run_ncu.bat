@echo off
REM -----------------------------------------------------------------
REM Batch file to launch Nsight Compute from your activated conda env
REM Profiles exactly the first extract_fields_kernel invocation.
REM -----------------------------------------------------------------

REM 1) (Optional) Change into your repo folder if you’re not already there
REM cd /d C:\3DMLGPU\p1\MicroDreamerOptimized

REM 2) Invoke Nsight Compute:
ncu ^
    --set full ^                       REM Collect the full default metric set (occupancy, mem BW, etc.)
    --kernel-name extract_fields_kernel ^  REM Only profile kernels named “extract_fields_kernel”
    --launch-count 1 ^                 REM Profile the first time that kernel is launched, then stop
    --target-processes all ^           REM Include any threads or subprocesses in the capture
    --export extract_fields_ncu_report.csv ^ REM Also write a human-readable CSV
    -- ^                               REM “--” separates ncu flags from the application command
    python main_profile.py ^           REM Your normal python invocation
         --config=configs/image_sai.yaml ^
         --input=test_data/05_objaverse_backpack_rgba.png ^
         --save_path=05_objaverse_backpack_rgba ^
         --profiling.enabled=true ^
         --profiling.mode=nvtx ^
         --profiling.scope=function ^
         --iters=20

REM 3) Pause so you can read any stdout/stderr before the window closes
pause