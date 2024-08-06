#!/bin/bash

#SBATCH -N 1
#SBATCH -c 12
#SBATCH -t 0-01:00:00
#SBATCH --mem=64G
#SBATCH -G a100:1
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
absoloute_path=
# the main python script name string to run the hitfunder model
script_name=

# the file path string to the input lst file containing the file paths to the data to run through the hitfinder model
path_to_input_lst_file=
# the name string of the input lst file containing the file paths to the data to run through the hitfinder model
lst_file_name=

# the string name of the model class to use for the hitfinder model
model_class=
# the file path string to the output lst files to save the hitfinder model results
path_to_training_results=

# the file path string to the model state dict file to use for the hitfinder model, or None if not using transfer learning
model_class_state_dict=

# the name string of the model state dict file to use for the hitfinder model
trained_model_state_dict=
# the file path string to the output lst files to save the hitfinder model results
path_to_trained_model_state_dict_output=

# integer value of the number of epochs to use for the hitfinder model
num_epochs=
# integer value of the batch size to use for the hitfinder model
batch_size=
# the string name of the optimizer to use for the hitfinder model
optimizer=
# the string name of the scheduler to use for the hitfinder model
scheduler=
# the string name of the criterion to use for the hitfinder model
criterion=
# float value of the learning rate to use for the hitfinder model
learning_rate=

# the string name of the camera length parameter to use for the hitfinder model, this can be a single string if from attribute manager, or a path if not using attribute manager or using master file
camera_length_parameter=
# the string name of the photon energy parameter to use for the hitfinder model, this can be a single string if from attribute manager, or a path if not using attribute manager or using master file
photon_energy_parameter=
# the string name of the hit parameter to use for the hitfinder model, this can be a single string if from attribute manager, or a path if not using attribute manager or using master file
hit_parameter=

# the string name of the transfer learning model state dict file to use for the hitfinder model, or None if not using transfer learning
transfer_learning=None

# Should the transform defined in prep_loaded_data class Data, be applied
apply_transform=True

# creates the transfer learning string if not None
if [ "$transfer_learning" != "None" ]; then
    transfer_learning="${path_to_trained_model_state_dict_output}${model_class_state_dict}"
fi

# Run the Python script with arguments
python ${absoloute_path}${script_name} -l ${path_to_input_lst_file}${lst_file_name} -m ${model_class} -o ${path_to_training_results} -d ${path_to_trained_model_state_dict_output}${trained_model_state_dict} -e ${num_epochs} -b ${batch_size} -op ${optimizer} -s ${scheduler} -c ${criterion} -lr ${learning_rate} -cl ${camera_length_parameter} -pe ${photon_energy_parameter} -pk ${hit_parameter} -tl ${transfer_learning} -at ${apply_transform}

