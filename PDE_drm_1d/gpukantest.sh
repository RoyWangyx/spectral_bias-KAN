#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -J "pde_gpu_combined"   # job name
#SBATCH --mail-user=roywang@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-9 

index=$SLURM_ARRAY_TASK_ID



# Run your command with the hyperparameter
python pde_combined.py $index 10


