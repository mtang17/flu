#!/bin/bash

#SBATCH --output=./slurm/output_%j.txt
#SBATCH --error=./slurm/error_%j.txt
#SBATCH --partition=NODES
#SBATCH --ntasks=2
#SBATCH --array=1-10
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=USER@INSTITUTION.EDU

python3 simulation_code.py ./inputs/main_results/base_Vax1.txt $SLURM_ARRAY_TASK_ID

