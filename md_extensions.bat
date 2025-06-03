@echo off

REM Run these steps from an x64 Native Tools Command Prompt as Admin
REM Make sure you are in a MicroDreamer directory with MicroDreamer already installed

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
echo pip install setuptools numpy ninja - may already be installed
echo ========================================

pip install setuptools numpy ninja

echo.
echo ========================================
echo Install the CUDA kernel(s) python setup_cuda_kernels.py build_ext --inplace
echo ========================================
python setup_cuda_kernels.py build_ext --inplace
