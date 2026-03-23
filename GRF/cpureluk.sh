#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=24   # number of processor cores (i.e. tasks)
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH -J "GRF_reluk"   # job name
#SBATCH --mail-user=roywang@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-23  

index=$SLURM_ARRAY_TASK_ID



# Run your command with the hyperparameter
python GRF_reluk.py $index 12



