#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -G a100:1
#SBATCH -t 0-00:10:00
#SBATCH --mem=128G
#SBATCH -p general
#SBATCH -q public 
#SBATCH -o /scratch/eseveret/cxls_hitfinder_joblogs/slurm.%j.out
#SBATCH -e /scratch/eseveret/cxls_hitfinder_joblogs/slurm.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user="eseveret@asu.edu"
#SBATCH --export=NONE

# Load necessary modules
module purge
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0 

# Activate the conda environment
source activate hitfinder_sol_env

# the file path string to the main python script to run the hitfunder model
path_to_script=
# the main python script name string to run the hitfunder model
script_name=

# the file path string to the input lst file containing the file paths to the data to run through the hitfinder model
path_to_input_lst_file=
# the name string of the input lst file containing the file paths to the data to run through the hitfinder model
input_lst_name=

# the string name of the model class to use for the hitfinder model
model_class=
# the file path string to the model state dict file to use for the hitfinder model
path_to_model_state_dict=
# the name string of the model state dict file to use for the hitfinder model
model_state_dict=

# the file path string to the output lst files to save the hitfinder model results
path_to_output_lst_files=

# the string name of the camera length parameter to use for the hitfinder model, this can be a single string if from attribute manager, or a path if not using attribute manager or using master file
camera_length_parameter=
# the string name of the photon energy parameter to use for the hitfinder model, this can be a single string if from attribute manager, or a path if not using attribute manager or using master file
photon_energy_parameter=

# integer value of the batch size to use for the hitfinder model
batch_size=

# string boolean value of whether to use multievent or not for the hitfinder model
multievent=

# the string name of the master file being used for metadata or None if not using a master file
master_file=None

# Run the Python script with arguments
python ${path_to_script}${script_name} -l ${path_to_input_lst_file}${input_lst_name} -m ${model_class} -d ${path_to_model_state_dict}${model_state_dict} -o ${path_to_output_lst_files} -cl ${camera_length_parameter} -pe ${photon_energy_parameter} -b ${batch_size} -me ${multievent} -mf ${master_file}

