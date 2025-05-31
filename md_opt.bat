@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Fixing corrupted numpy installation...
echo ========================================
del /q /f C:\conda\pynerf2\Lib\site-packages\-umpy* 2>nul
rmdir /s /q C:\conda\pynerf2\Lib\site-packages\numpy 2>nul
rmdir /s /q C:\conda\pynerf2\Lib\site-packages\numpy-* 2>nul

echo.
echo ========================================
echo Clearing pip cache...
echo ========================================
pip cache purge

echo.
echo ========================================
echo Uninstalling any previously installed torch stack...
echo ========================================
pip uninstall -y torch torchvision torchaudio

echo.
echo ========================================
echo Installing numpy cleanly...
echo ========================================
pip install --force-reinstall numpy

echo.
echo ========================================
echo Installing PyTorch stack with CUDA 12.4 support...
echo ========================================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo ========================================
echo Verifying torch is available...
echo ========================================
python -c "import torch; print('Verified torch installation:', torch.__version__)"
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: PyTorch is not installed correctly.
    exit /b 1
)

echo.
echo ========================================
echo Installing Python dependencies from requirements.txt...
echo ========================================
pip install -r requirements.txt

echo.
echo ========================================
echo Adding path to cl.exe to PATH
echo ========================================
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x86;%PATH%"

echo.
echo ========================================
echo Installing build tools (setuptools, wheel, cmake)...
echo ========================================
pip install setuptools wheel cmake

echo.
echo ========================================
echo Installing diff-gaussian-rasterization with no build isolation...
echo ========================================
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install --no-build-isolation ./diff-gaussian-rasterization

echo.
echo ========================================
echo Installing simple-knn...
echo ========================================
git clone https://github.com/ashawkey/simple-knn
pip install --no-build-isolation ./simple-knn

echo.
echo ========================================
echo Installing additional Git-based dependencies...
echo ========================================
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install git+https://github.com/ashawkey/kiuikit/
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream

echo.
echo ========================================
echo Final numpy reinstall (in case it was overwritten)
echo ========================================
pip install --force-reinstall numpy

echo.
echo ========================================
echo Verifying installations...
echo ========================================
python -c "import torch; import torchvision; import torchaudio; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('Torchaudio:', torchaudio.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); print('cuDNN:', torch.backends.cudnn.version()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Checking CUDA compatibility...
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.version.cuda == '12.4' else 1)"
IF ERRORLEVEL 1 (
    echo.
    echo ========================================
    echo ERROR: CUDA 12.4 not detected or GPU is not available!
    echo ========================================
    exit /b 1
)

echo.
echo ========================================
echo Test Command preceeded by set PATH command
echo ========================================
echo You may need to reset the path to cl.exe before running the application, do as follows:
echo set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x86;%PATH%"
echo python main.py --config configs/image_sai.yaml input=test_data/05_objaverse_backpack_rgba.png save_path=05_objaverse_backpack_rgba

echo.
echo ========================================
echo Installation and verification successful.
echo ========================================
pause
endlocal

REM python main.py --config configs/image_sai.yaml input=test_data/05_objaverse_backpack_rgba.png save_path=05_objaverse_backpack_rgba
