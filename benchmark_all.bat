@echo off
setlocal enabledelayedexpansion

:: Use UTF-8 to avoid Unicode errors
chcp 65001 >nul

:: Detect “--profile” flag
set PROFILE_MODE=0
if "%1"=="--profile" set PROFILE_MODE=1

:: Bench settings
set INPUT=test_data/05_objaverse_backpack_rgba.png
set SAVE_PATH=05_objaverse_backpack_rgba
set CONFIG=configs/image_sai.yaml
set ITERS=20
set LOGDIR=benchmarks
if not exist %LOGDIR% mkdir %LOGDIR%

:: Timestamp
for /f %%i in ('wmic os get localdatetime ^| find "."') do set DTS=%%i
set DTS=%DTS:~0,8%_%DTS:~8,6%
set SUMMARY=%LOGDIR%\timing_summary_%DTS%.txt
echo Benchmark Timing Summary > "%SUMMARY%"

echo [INFO] PROFILE_MODE=%PROFILE_MODE%

for %%C in (baseline cuda_gauss_only cuda_extract_only full_cuda) do (
    echo ---------------------------------- >> "%SUMMARY%"
    echo Running: %%C
    echo Running: %%C >> "%SUMMARY%"

    :: Corrected IF cascade for the two flags
    if "%%C"=="baseline" (
        set USE_CUDA_GAUSS=0
        set USE_CUDA_EXTRACT=0
    ) else if "%%C"=="cuda_gauss_only" (
        set USE_CUDA_GAUSS=1
        set USE_CUDA_EXTRACT=0
    ) else "%%C"=="cuda_extract_no_gauss" (
        set USE_CUDA_GAUSS=0
        set USE_CUDA_EXTRACT=1
    )

    set RUN_LABEL=%%C
    set LOGFILE=%LOGDIR%\%%C_%DTS%.log

    :: Echo config
    echo [INFO] USE_CUDA_GAUSS=!USE_CUDA_GAUSS! >> "!LOGFILE!"
    echo [INFO] USE_CUDA_EXTRACT=!USE_CUDA_EXTRACT! >> "!LOGFILE!"
    echo [INFO] RUN_LABEL=!RUN_LABEL! >> "!LOGFILE!"

    if "!PROFILE_MODE!"=="1" (
        echo [INFO] Profiling with Nsight Systems… >> "!LOGFILE!"
        echo [INFO] Running: nsys profile --trace=cuda,nvtx --report summary --report-output=%LOGDIR%\nsys_%%C_%DTS%_summary --output=%LOGDIR%\nsys_%%C_%DTS% >> "!LOGFILE!"

        REM 1) Run and capture a .nsys-rep plus a summary CSV
        nsys profile ^
            --trace=cuda,nvtx ^
            --report summary ^
            --report-output "%LOGDIR%\nsys_%%C_%DTS%_summary" ^
            --output "%LOGDIR%\nsys_%%C_%DTS%" ^
            --force-overwrite=true ^
            python main_profile.py ^
                --config=!CONFIG! ^
                --input=!INPUT! ^
                --save_path=!SAVE_PATH! ^
                --profiling.enabled=true ^
                --profiling.mode=nvtx ^
                --profiling.scope=function ^
                --iters=!ITERS! >> "!LOGFILE!" 2>&1

        REM 2) Append the kernel summary into our log
        echo. >> "!LOGFILE!"
        echo [INFO] Kernel timing summary: >> "!LOGFILE!"
        nsys stats ^
            --report summary ^
            "%LOGDIR%\nsys_%%C_%DTS%.nsys-rep" >> "!LOGFILE!" 2>&1

    ) else (
        echo [INFO] Running without profiling… >> "!LOGFILE!"

        REM measure total runtime in seconds via PowerShell
        echo [INFO] Command: python main_profile.py --config !CONFIG! --input !INPUT! --save_path !SAVE_PATH! --profiling.enabled false --iters !ITERS! >> "!LOGFILE!"
        for /f "usebackq tokens=*" %%t in (`powershell -Command "(Measure-Command { python main_profile.py --config %CONFIG% --input %INPUT% --save_path %SAVE_PATH% --profiling.enabled false --iters %ITERS% }).TotalSeconds"`) do set DURATION=%%t

        echo !RUN_LABEL!: !DURATION! seconds >> "!LOGFILE!"
        echo !RUN_LABEL!: !DURATION! seconds >> "%SUMMARY%"
    )

    echo Done: %%C >> "%SUMMARY%"
    echo [INFO] Output saved to: !LOGFILE!
)

echo ----------------------------------
echo All runs complete. Summary at: %SUMMARY%
pause