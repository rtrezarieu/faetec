@echo off
REM Transfer files in the data directory to the cluster, excluding specified files and directories

REM
set LOCAL_DIR="c:/Users/raphael.trezarieu/OneDrive - Setec/Documents/PFE/Code/grm-faenet-main/data/"
set REMOTE_USER="raphtrez"
set REMOTE_HOST="satori-login-001.mit.edu"
set REMOTE_DIR="~/GRM-FAENET/data/"
set IGNORE_FILE="../.scpignore"
set SSH_KEY="~/.ssh/id_rsa"

REM Change to the local directory
cd %LOCAL_DIR%

REM Use rsync to transfer files in the data directory, excluding those specified in .scpignore
REM Note: This command runs in WSL
@REM wsl rsync -av --exclude-from=%IGNORE_FILE% . %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%
wsl rsync -av --exclude-from=%IGNORE_FILE% -e "ssh -i %SSH_KEY%" . %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%