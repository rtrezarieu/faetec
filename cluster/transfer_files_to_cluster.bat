@echo off
REM Transfer files to the cluster, excluding specified files and directories

REM Define variables
set LOCAL_DIR="c:/Users/raphael.trezarieu/OneDrive - Setec/Documents/PFE/Code/grm-faenet-main/"
set REMOTE_USER="raphtrez"
set REMOTE_HOST="satori-login-001.mit.edu"
set REMOTE_DIR="~/GRM-FAENET/"
set IGNORE_FILE=".scpignore"

REM Change to the local directory
cd %LOCAL_DIR%

REM Use rsync to transfer files, excluding those specified in .scpignore
REM Note: This command runs in WSL
wsl rsync -av --exclude-from=%IGNORE_FILE% . %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%