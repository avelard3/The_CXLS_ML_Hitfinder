#!/bin/bash

#SBATCH -N 1 
#SBATCH -c 12
#SBATCH -t 0-01:00:00
#SBATCH --mem=64G
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
module load cuda-12.5.0-gcc-12.1.0 

source activate hitfinder_sol_env

#current data is epix10k multipanel detector
absoloute_path='/scratch/avelard3/The_CXLS_ML_Hitfinder/src/'
script_name='train_and_evaluate_hitfinder.py'

path_to_input_lst_file='/scratch/avelard3/'
lst_file_name='epix10kdata.lst' #list file taht contains data paths and files

model_class='Binary_Classification_With_Parameters'
path_to_training_results='/home/avelard3/hitfinder_output_files/train_model_output'

model_class_state_dict='hitfinder_model_7.pt'

trained_model_state_dict='transform_test.pt'
path_to_trained_model_state_dict_output='/home/avelard3/hitfinder_models/'

num_epochs=5
batch_size=10
optimizer='Adam'
scheduler='ReduceLROnPlateau'
criterion='BCEWithLogitsLoss'
learning_rate=0.0001

camera_length_parameter='camera_length'
photon_energy_parameter='photon_energy'
hit_parameter='hit_parameter'

transfer_learning=None
apply_transform=False

# string boolean value of whether to use multievent or not for the hitfinder model
multievent=True

# the string name of the master file being used for metadata or None if not using a master file
master_file=None


if [ "$transfer_learning" != "None" ]; then
    transfer_learning="${path_to_trained_model_state_dict_output}${model_class_state_dict}"
fi

python ${absoloute_path}${script_name} -l ${path_to_input_lst_file}${lst_file_name} -m ${model_class} -o ${path_to_training_results} -d ${path_to_trained_model_state_dict_output}${trained_model_state_dict} -e ${num_epochs} -b ${batch_size} -op ${optimizer} -s ${scheduler} -c ${criterion} -lr ${learning_rate} -cl ${camera_length_parameter} -pe ${photon_energy_parameter} -pk ${hit_parameter} -tl ${transfer_learning} -at ${apply_transform} -me ${multievent} -mf ${master_file}