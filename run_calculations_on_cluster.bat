@echo off
REM Launch calculations on the cluster via SSH

set REMOTE_USER="raphtrez"
set REMOTE_HOST="satori-login-001.mit.edu"
set REMOTE_SCRIPT="~/GRM-FAENET/run_calculations.sh"
set SSH_KEY="~/.ssh/id_rsa"

REM Use SSH to connect to the remote cluster and execute the script
wsl ssh -i %SSH_KEY% %REMOTE_USER%@%REMOTE_HOST% "bash %REMOTE_SCRIPT%"