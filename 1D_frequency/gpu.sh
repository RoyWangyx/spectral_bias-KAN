#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -J "1D_frequency_gpu"   # job name
#SBATCH --mail-user=roywang@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-3  

index=$SLURM_ARRAY_TASK_ID



# Run your command with the hyperparameter
python 1D_frequency.py $index 4


