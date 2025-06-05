@echo off

REM Run these steps from an x64 Native Tools Command Prompt as Admin
REM Make sure you are in a MicroDreamer directory with MicroDreamer already installed

REM ------------------------------------------------
REM 1) Verify that we are inside the "pynerf2" Conda env
REM ------------------------------------------------

REM Check if CONDA_DEFAULT_ENV is defined at all
IF NOT DEFINED CONDA_DEFAULT_ENV (
    echo.
    echo ======================================================
    echo ERROR: You are not inside any Conda environment.
    echo Please run:
    echo   conda activate C:\conda\pynerf2
    echo before running this script.
    echo ======================================================
    exit /b 1
)

echo.
echo ========================================
echo Make sure you are in a MicroDreamer directory with MicroDreamer already installed
echo ========================================
echo ========================================


echo.
echo ========================================
echo set DISTUTILS_USE_SDK=1
echo ========================================
set DISTUTILS_USE_SDK=1

echo.
echo ========================================
echo pip install setuptools numpy ninja (note that these may already be installed)
echo ========================================

pip install setuptools numpy ninja

echo.
echo ========================================
echo Install the CUDA kernel(s) python setup_cuda_kernels.py build_ext --inplace
echo ========================================
python setup_cuda_kernels.py build_ext --inplace
