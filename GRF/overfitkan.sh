#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=12   # number of processor cores (i.e. tasks)
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -J "overfit_KAN"   # job name
#SBATCH --mail-user=roywang@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-11  

index=$SLURM_ARRAY_TASK_ID



# Run your command with the hyperparameter
python GRF_KAN_overfit.py $index 12