#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -J "adam"   # job name
#SBATCH --mail-user=roywang@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-35

index=$SLURM_ARRAY_TASK_ID



# Run your command with the hyperparameter
python drm_adam.py $index 36


