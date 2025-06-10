@echo off
setlocal enabledelayedexpansion

:: Use UTF-8 to avoid Unicode issues
chcp 65001 >nul

:: Check for optional --profile flag
set PROFILE_MODE=0
if "%1"=="--profile" set PROFILE_MODE=1

:: Run parameters
set INPUT=test_data/05_objaverse_backpack_rgba.png
set SAVE_PATH=05_objaverse_backpack_rgba
set CONFIG=configs/image_sai.yaml
set ITERS=20

:: Log folder
set LOGDIR=benchmarks
if not exist %LOGDIR% mkdir %LOGDIR%

:: Timestamp
for /f %%i in ('wmic os get localdatetime ^| find "."') do set DTS=%%i
set DTS=%DTS:~0,8%_%DTS:~8,6%
set SUMMARY=%LOGDIR%\timing_summary_%DTS%.txt
echo Benchmark Timing Summary > "%SUMMARY%"

echo [INFO] PROFILE_MODE=%PROFILE_MODE%

:: Loop over 3 configs
for %%C in (baseline cuda_gauss_only cuda_extract_only) do (
    echo ---------------------------------- >> "%SUMMARY%"
    echo Running: %%C
    echo Running: %%C >> "%SUMMARY%"

    if "%%C"=="baseline" (
        set USE_CUDA_GAUSS=0
        set USE_CUDA_EXTRACT=0
    ) else if "%%C"=="cuda_gauss_only" (
        set USE_CUDA_GAUSS=1
        set USE_CUDA_EXTRACT=0
    ) else if "%%C"=="cuda_extract_only" (
        set USE_CUDA_GAUSS=0
        set USE_CUDA_EXTRACT=1
    )

    set RUN_LABEL=%%C
    set LOGFILE=%LOGDIR%\%%C_%DTS%.log

    echo [INFO] USE_CUDA_GAUSS=!USE_CUDA_GAUSS! >> "!LOGFILE!"
    echo [INFO] USE_CUDA_EXTRACT=!USE_CUDA_EXTRACT! >> "!LOGFILE!"
    echo [INFO] RUN_LABEL=!RUN_LABEL! >> "!LOGFILE!"

    if "!PROFILE_MODE!"=="1" (
        echo [INFO] Profiling with Nsight Systems... >> "!LOGFILE!"
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
                --profiling.scope=function >> "!LOGFILE!" 2>&1

        echo. >> "!LOGFILE!"
        echo [INFO] Kernel timing summary: >> "!LOGFILE!"
        nsys stats ^
            --report summary ^
            "%LOGDIR%\nsys_%%C_%DTS%.nsys-rep" >> "!LOGFILE!" 2>&1

    ) else (
        echo [INFO] Running without profiling... >> "!LOGFILE!"
        set START=%TIME%
        python main_profile.py --config %CONFIG% --input %INPUT% --save_path %SAVE_PATH% --profiling.enabled false --iters %ITERS%
        set END=%TIME%
        echo !RUN_LABEL!: start=%START% end=%END% >> "!LOGFILE!"
        echo !RUN_LABEL!: start=%START% end=%END% >> "%SUMMARY%"
    )

    echo Done: %%C >> "%SUMMARY%"
    echo [INFO] Output saved to: !LOGFILE!
)

echo ----------------------------------
echo All runs complete. Summary at: %SUMMARY%
pause