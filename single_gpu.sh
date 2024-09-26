#!/bin/bash
#SBATCH -J FAENet_struct
#SBATCH -o logs/FAENet_struct_%j.out
#SBATCH -e logs/FAENet_struct_%j.err

#SBATCH --mail-user=raphtrez@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=opence_recent
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

# Change to the directory where the data files are located
cd ~/GRM-FAENET

# Create logs directory if it doesn't exist
mkdir -p logs

echo " Run started at:- "
date

# python ~/GRM-FAENET/main.py debug=True
python ~/GRM-FAENET/main.py equivariance=frame_averaging fa_type=stochastic

echo "Run completed at:- "
date
