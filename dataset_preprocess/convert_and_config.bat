@echo off

set LOCAL_DIR=c:/Users/raphael.trezarieu/Documents/Code/faetec/
set VENV_PYTHON=c:/Users/raphael.trezarieu/Documents/Code/faetec/.venv/Scripts/python.exe
set DATASET_NAME="blocks_unique_random_last_floor_2x2x2_10"

cd %LOCAL_DIR%

%VENV_PYTHON% dataset_preprocess/dataset_convertor.py %DATASET_NAME%

if %errorlevel% neq 0 (
    echo Error running dataset_convertor.py
    exit /b %errorlevel%
)

%VENV_PYTHON% dataset_preprocess/config_generator.py %DATASET_NAME%

if %errorlevel% neq 0 (
    echo Error running config_generator.py
    exit /b %errorlevel%
)

echo Scripts executed successfully.