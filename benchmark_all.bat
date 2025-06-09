@echo off
setlocal enabledelayedexpansion

:: Create log folder
set LOGDIR=benchmarks
if not exist %LOGDIR% mkdir %LOGDIR%

:: Get date-time stamp
for /f %%i in ('wmic os get localdatetime ^| find "."') do set DTS=%%i
set DTS=%DTS:~0,8%_%DTS:~8,6%

:: Path to timing summary
set SUMMARY=%LOGDIR%\timing_summary_%DTS%.txt
echo Benchmark Timing Summary > %SUMMARY%

:: Run Scenarios
set CFG_LIST=baseline cuda_gauss_only cuda_extract_only full_cuda

for %%C in (%CFG_LIST%) do (
    echo ---------------------------------- >> %SUMMARY%
    echo Running: %%C
    echo Running: %%C >> %SUMMARY%
    set "USE_CUDA_GAUSS=0"
    set "USE_CUDA_EXTRACT=0"

    if "%%C"=="cuda_gauss_only"  set "USE_CUDA_GAUSS=1"
    if "%%C"=="cuda_extract_only" set "USE_CUDA_EXTRACT=1"
    if "%%C"=="full_cuda" (
        set "USE_CUDA_GAUSS=1"
        set "USE_CUDA_EXTRACT=1"
    )

    set "RUN_LABEL=%%C"
    set "LOGFILE=%LOGDIR%\%%C_%DTS%.log"

    set USE_CUDA_GAUSS=!USE_CUDA_GAUSS!
    set USE_CUDA_EXTRACT=!USE_CUDA_EXTRACT!
    set RUN_LABEL=!RUN_LABEL!

    echo [INFO] USE_CUDA_GAUSS=!USE_CUDA_GAUSS! >> !LOGFILE!
    echo [INFO] USE_CUDA_EXTRACT=!USE_CUDA_EXTRACT! >> !LOGFILE!
    echo [INFO] RUN_LABEL=!RUN_LABEL! >> !LOGFILE!

    :: Export env vars for Python
    setx USE_CUDA_GAUSS !USE_CUDA_GAUSS! >nul
    setx USE_CUDA_EXTRACT !USE_CUDA_EXTRACT! >nul
    setx RUN_LABEL !RUN_LABEL! >nul

    python main_profile.py --config configs/image_sai.yaml >> !LOGFILE!

    echo Done: %%C >> %SUMMARY%
)

echo ----------------------------------
echo All runs complete. Summary at: %SUMMARY%
notepad %SUMMARY%
