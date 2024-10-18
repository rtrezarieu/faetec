@echo off
REM Transfer files to the cluster, excluding specified files and directories

REM Define variables
set LOCAL_DIR="c:/Users/raphael.trezarieu/OneDrive - Setec/Documents/PFE/Code/grm-faenet-main/"
set REMOTE_USER="raphtrez"
set REMOTE_HOST="satori-login-001.mit.edu"
set REMOTE_DIR="~/GRM-FAENET/"
set IGNORE_FILE=".scpignore"
set SSH_KEY="c:/Users/raphael.trezarieu/.ssh/id_rsa"

REM Convert Windows path to WSL path
for /f "tokens=*" %%i in ('wsl wslpath "%SSH_KEY%"') do set WSL_SSH_KEY=%%i

REM Change to the local directory
cd %LOCAL_DIR%

REM Use rsync to transfer files, excluding those specified in .scpignore
REM Note: This command runs in WSL
@REM wsl rsync -av --exclude-from=%IGNORE_FILE% . %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%
wsl rsync -av --exclude-from=%IGNORE_FILE% -e "ssh -i %WSL_SSH_KEY%" . %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%