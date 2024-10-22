#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -G a100:1
#SBATCH -t 0-00:10:00
#SBATCH --mem=128G
#SBATCH -p general
#SBATCH -q public 
#SBATCH -o /scratch/avelard3/cxls_hitfinder_joblogs/slurm.%j.out
#SBATCH -e /scratch/avelard3/cxls_hitfinder_joblogs/slurm.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user="avelard3@asu.edu"
#SBATCH --export=NONE



# Load necessary modules
module purge
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0 

# Activate the conda environment
source activate hitfinder_sol_env

# the file path string to the main python script to run the hitfunder model
path_to_script='/scratch/avelard3/The_CXLS_ML_Hitfinder/src/'
# the main python script name string to run the hitfunder model
script_name='run_hitfinder_model.py'

# the file path string to the input lst file containing the file paths to the data to run through the hitfinder model
path_to_input_lst_file='/scratch/avelard3/'
# the name string of the input lst file containing the file paths to the data to run through the hitfinder model
input_lst_name='test_VDS_data.lst'

# the string name of the model class to use for the hitfinder model
model_class='Binary_Classification_With_Parameters'
# the file path string to the model state dict file to use for the hitfinder model
path_to_model_state_dict='/home/eseveret/Annelise/'
# the name string of the model state dict file to use for the hitfinder model
model_state_dict='hitfinder_model_7.pt'

# the file path string to the output lst files to save the hitfinder model results
path_to_output_lst_files='/home/avelard3/hitfinder_output_files/run_model_output'

# the string name of the camera length parameter to use for the hitfinder model, this can be a single string if from attribute manager, or a path if not using attribute manager or using master file
camera_length_parameter='camera_length'
# the string name of the photon energy parameter to use for the hitfinder model, this can be a single string if from attribute manager, or a path if not using attribute manager or using master file
photon_energy_parameter='photon_energy'

# integer value of the batch size to use for the hitfinder model
batch_size=10

# string boolean value of whether to use multievent or not for the hitfinder model
multievent=True

# the string name of the master file being used for metadata or None if not using a master file
master_file=None

# Run the Python script with arguments
python ${path_to_script}${script_name} -l ${path_to_input_lst_file}${input_lst_name} -m ${model_class} -d ${path_to_model_state_dict}${model_state_dict} -o ${path_to_output_lst_files} -cl ${camera_length_parameter} -pe ${photon_energy_parameter} -b ${batch_size} -me ${multievent} -mf ${master_file}

