#!/bin/bash

#SBATCH -N 1 
#SBATCH -c 12
# max time that the job can run for (currently set to 1 day)
#SBATCH -t 1-00:00:00

# specify which supercomputer nodes will be used to run the model
#SBATCH --mem=64G
#SBATCH -G h100:1
#SBATCH -p general
#SBATCH -q grp_cxfel 

# where to save output and error files
#SBATCH -o /scratch/avelard3/cxls_hitfinder_joblogs/slurm.%j.out
#SBATCH -e /scratch/avelard3/cxls_hitfinder_joblogs/slurm.%j.err

# send email to this address when job ends
#SBATCH --mail-type=END
#SBATCH --mail-user="avelard3@asu.edu"
#SBATCH --export=NONE

# activate Hitfinder environment **MUST USE MAMBA IN SOL**
module purge
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0 

source activate hitfinder_env_try8
export HDF5_PLUGIN_PATH=/home/avelard3/.conda/envs/hitfinder_env_try8/lib/python3.11/site-packages/hdf5plugin/plugins


#current data is epix10k multipanel detector
absoloute_path='/path/to/script/' # Location of main python script. Ex: '/scratch/avelard3/The_CXLS_ML_Hitfinder/src/'
script_name='hyperparameter_tuning.py' # Name of main python script being run

path_to_input_lst_file='/path/to/script/' # Location of list. Ex:'/scratch/avelard3/sbatch_scripts/specific_file_lists/'
lst_file_name='many_sim_few_real.lst' # list (.lst) file that contains data paths and files

model_class='Optuna_Simple_CNN' # Model type in models.py (Normally Optunas_Simple_CNN when training)
path_to_training_results='/home/avelard3/hitfinder_output_files/train_model_output' # !FIXME Is this necessary???

model_class_state_dict=None

trained_model_state_dict='first_try_with_masters.pt'
path_to_trained_model_state_dict_output='/home/avelard3/hitfinder_models/'

# Permanent hyperparameters>>
batch_size=50
optimizer='Adam'
scheduler='ReduceLROnPlateau'
criterion='BCEWithLogitsLoss'

path_to_geom=None

transfer_learning=None


if [ "$transfer_learning" != "None" ]; then
    transfer_learning="${path_to_trained_model_state_dict_output}${model_class_state_dict}"
fi

# Range "lower_val upper_val" for possible values that optuna will try
#the formatting for the numbers is so that they will be easily transformed into tuples in the python script
epoch_range="5 100"
learning_rate_range="0.00001 0.01"
lr_param_patience_range="3 100"
lr_param_threshold_range="0.001 0.1" 

conv_channel_size_range="2 8"
conv_kernel_size_range="3 10" 
num_linear_dropout_layers_range="1 3"
linear_layer_size_range="2 8" #think about it -ee
dropout_probability_range="0.3 0.8"

# Name of Optuna study and number of trials of Optuna
optuna_study_name="the-cxls-ml-hitfinder-trial1_dec19.2"
num_trials=5

python ${absoloute_path}${script_name} \
    -sn ${optuna_study_name} \
    -nt ${num_trials}\
    -l ${path_to_input_lst_file}${lst_file_name} \
    -m ${model_class} \
    -b ${batch_size} \
    -op ${optimizer} \
    -s ${scheduler} \
    -c ${criterion} \
    -tl ${transfer_learning} \
    -er ${epoch_range} \
    -lrr ${learning_rate_range} \
    -lrpp ${lr_param_patience_range} \
    -lrpt ${lr_param_threshold_range} \
    -ccs ${conv_channel_size_range} \
    -cks ${conv_kernel_size_range} \
    -ldl ${num_linear_dropout_layers_range} \
    -lls ${linear_layer_size_range} \
    -dop ${dropout_probability_range} \
    -g ${path_to_geom}

