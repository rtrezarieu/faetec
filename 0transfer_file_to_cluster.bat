@echo off
REM Transfer files to the cluster - Templates
REM scp my_file.py raphtrez@satori-login-001.mit.edu:~/GRM-FAENET/
REM scp -r my_directory raphtrez@satori-login-001.mit.edu:~/GRM-FAENET/

REM To execute first from VSCode terminal
REM .\transfer_files_to_cluster.bat
cd "c:/Users/raphael.trezarieu/OneDrive - Setec/Documents/PFE/Code/grm-faenet-main/"
scp dataset_convertor.py raphtrez@satori-login-001.mit.edu:~/GRM-FAENET/