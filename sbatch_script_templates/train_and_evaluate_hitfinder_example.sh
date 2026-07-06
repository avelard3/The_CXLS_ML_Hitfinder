#!/bin/bash

#SBATCH -N 1 
#SBATCH -c 12
# max time that the job can run for
#SBATCH -t 0-03:00:00

# specify which supercomputer nodes will be used to run the model
#SBATCH --mem=64G
#SBATCH -G h100:1
#SBATCH -p general
#SBATCH -q grp_cxfel 

# where to save output and error files
#SBATCH -o /scratch/ASURITE/cxls_hitfinder_joblogs/slurm.%j.out
#SBATCH -e /scratch/ASURITE/cxls_hitfinder_joblogs/slurm.%j.err

# send email to this address when job ends
#SBATCH --mail-type=END
#SBATCH --mail-user="ASURITE@asu.edu"
#SBATCH --export=NONE

# activate Hitfinder environment **MUST USE MAMBA IN SOL**
module purge
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0 

source activate hitfinder_env_try8
export HDF5_PLUGIN_PATH=/home/avelard3/.conda/envs/hitfinder_env_try8/lib/python3.11/site-packages/hdf5plugin/plugins


# Main python script being run
absoloute_path='/path/to/script/' # Location of main python script. Ex: '/scratch/avelard3/The_CXLS_ML_Hitfinder/src/'
script_name='train_and_evaluate_hitfinder.py' # Name of main python script being run

# List of .h5 files (see documentation for format requirements of .lst file) 
path_to_input_lst_file='/path/to/script/' # Location of list. Ex:'/scratch/avelard3/sbatch_scripts/specific_file_lists/'
lst_file_name='many_sim_few_real.lst' # list (.lst) file that contains data paths and files

model_class='CNN_with_Optunas_Best' # Model type in models.py (Normally CNN_with_Optunas_Best when training)

path_to_training_results='/home/avelard3/hitfinder_output_files/train_model_output' # location for model that was created

# Name of transfer learning model
model_class_state_dict='CNN_with_Optunas_simreal_june3.pt'
# !FIXME:WHAT THE HECK IS THIS SECTION I DONT UNDERSTAND
trained_model_state_dict='CNN_with_Optunas_simreal_june3.pt'
path_to_trained_model_state_dict_output='/home/avelard3/hitfinder_models/'

# Basic hyperparameters for training
num_epochs=20
learning_rate=0.001

batch_size=8
optimizer='Adam'
scheduler='ReduceLROnPlateau'
criterion='BCEWithLogitsLoss'

path_to_geom=None

# this suposedly shows where the data is in the h5 files, need to change it if the data happens to be in differently named h5

transfer_learning=None


if [ "$transfer_learning" != "None" ]; then
    transfer_learning="${path_to_trained_model_state_dict_output}${model_class_state_dict}"
fi

python ${absoloute_path}${script_name} \
    -l ${path_to_input_lst_file}${lst_file_name} \
    -m ${model_class} \
    -o ${path_to_training_results} \
    -d ${path_to_trained_model_state_dict_output}${trained_model_state_dict} \
    -e ${num_epochs} \
    -b ${batch_size} \
    -op ${optimizer} \
    -s ${scheduler} \
    -c ${criterion} \
    -lr ${learning_rate} \
    -tl ${transfer_learning} \
    -g ${path_to_geom}