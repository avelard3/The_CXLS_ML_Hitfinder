#!/bin/bash

#SBATCH -N 1 
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --mem=16G
#SBATCH -G a100:1
#SBATCH -p general
#SBATCH -q public 
#SBATCH -o /scratch/avelard3/cxls_hitfinder_joblogs/slurm.%j.out
#SBATCH -e /scratch/avelard3/cxls_hitfinder_joblogs/slurm.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user="avelard3@asu.edu"
#SBATCH --export=NONE

module purge
module load mamba/latest

module load cuda-11.8.0-gcc-12.1.0

mamba remove -n hitfinder_sol_env --all
mamba clean --all

mamba env create -f /scratch/avelard3/environment_v2.yaml