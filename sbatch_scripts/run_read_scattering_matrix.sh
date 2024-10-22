#!/bin/bash

#SBATCH -N 1 
#SBATCH -c 4
#SBATCH -t 0-00:05:00
#SBATCH --mem=32G
#SBATCH -p general
#SBATCH -q public 
#SBATCH -o /scratch/avelard3/cxls_hitfinder_joblogs/slurm.%j.out
#SBATCH -e /scratch/avelard3/cxls_hitfinder_joblogs/slurm.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user="avelard3@asu.edu"
#SBATCH --export=NONE

module purge
module load mamba/latest
#run this on a cpu not a gpu
module load cuda-11.8.0-gcc-12.1.0
source activate hitfinder_sol_env

python /scratch/avelard3/The_CXLS_ML_Hitfinder/src/lib/read_scattering_matrix.py