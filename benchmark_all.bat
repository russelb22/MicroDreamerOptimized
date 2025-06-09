@echo off
setlocal enabledelayedexpansion

:: Use UTF-8 to avoid UnicodeEncodeError
chcp 65001 >nul

:: Check for optional --profile flag
set PROFILE_MODE=0
if "%1"=="--profile" (
    set PROFILE_MODE=1
)

:: Set run parameters
set INPUT=test_data/05_objaverse_backpack_rgba.png
set SAVE_PATH=05_objaverse_backpack_rgba
set CONFIG=configs/image_sai.yaml
set ITERS=20

:: Create output folder
set LOGDIR=benchmarks
if not exist %LOGDIR% mkdir %LOGDIR%

:: Timestamp
for /f %%i in ('wmic os get localdatetime ^| find "."') do set DTS=%%i
set DTS=%DTS:~0,8%_%DTS:~8,6%
set SUMMARY=%LOGDIR%\timing_summary_%DTS%.txt
echo Benchmark Timing Summary > %SUMMARY%

echo [INFO] PROFILE_MODE=%PROFILE_MODE%

:: Loop over 4 configs
for %%C in (baseline cuda_gauss_only cuda_extract_only full_cuda) do (
    echo ---------------------------------- >> %SUMMARY%
    echo Running: %%C
    echo Running: %%C >> %SUMMARY%

    if "%%C"=="baseline" (
        set USE_CUDA_GAUSS=0
        set USE_CUDA_EXTRACT=0
    ) else if "%%C"=="cuda_gauss_only" (
        set USE_CUDA_GAUSS=1
        set USE_CUDA_EXTRACT=0
    ) else if "%%C"=="cuda_extract_only" (
        set USE_CUDA_GAUSS=0
        set USE_CUDA_EXTRACT=1
    ) else (
        set USE_CUDA_GAUSS=1
        set USE_CUDA_EXTRACT=1
    )

    set RUN_LABEL=%%C
    set LOGFILE=%LOGDIR%\%%C_%DTS%.log

    echo [INFO] USE_CUDA_GAUSS=!USE_CUDA_GAUSS! >> !LOGFILE!
	echo [INFO] USE_CUDA_GAUSS=!USE_CUDA_GAUSS!
    echo [INFO] USE_CUDA_EXTRACT=!USE_CUDA_EXTRACT! >> !LOGFILE!
	echo [INFO] USE_CUDA_EXTRACT=!USE_CUDA_EXTRACT!
    echo [INFO] RUN_LABEL=!RUN_LABEL! >> !LOGFILE!

    if "!PROFILE_MODE!"=="1" (
        echo [INFO] Profiling with Nsight Systems... >> !LOGFILE!
        echo [INFO] Command: nsys profile --trace=cuda,nvtx --output=%LOGDIR%\nsys_%%C_%DTS% ... >> !LOGFILE!
        echo [INFO] Output saved to: !LOGFILE!

        nsys profile ^
		--trace=cuda,nvtx ^
		--sample=none ^
		--output=%LOGDIR%\nsys_%%C_%DTS% ^
		--force-overwrite=true ^
		python main_profile.py ^
		--config=!CONFIG! ^
		--input=!INPUT! ^
		--save_path=!SAVE_PATH! ^
		--profiling.enabled=true ^
		--profiling.mode=nvtx ^
		--profiling.scope=function ^
		--iters=!ITERS! ^
		--summary_path="!SUMMARY!" >> !LOGFILE! 2>&1
    ) else (
        echo [INFO] Running without profiling... >> !LOGFILE!

        :: Set env vars in current shell so python picks them up
        set USE_CUDA_GAUSS=!USE_CUDA_GAUSS!
        set USE_CUDA_EXTRACT=!USE_CUDA_EXTRACT!
        set RUN_LABEL=!RUN_LABEL!

        echo [INFO] Command: python main_profile.py --config !CONFIG! --input !INPUT! --save_path !SAVE_PATH! --profiling.enabled false --iters !ITERS! --summary_path "!SUMMARY!" >> !LOGFILE!
        python -X utf8 main_profile.py --config !CONFIG! --input !INPUT! --save_path !SAVE_PATH! --profiling.enabled false --iters !ITERS! --summary_path "!SUMMARY!" >> !LOGFILE! 2>&1
    )

    echo Done: %%C >> %SUMMARY%
	echo [INFO] Output saved to: !LOGFILE!
)

echo ----------------------------------
echo All runs complete. Summary at: %SUMMARY%
