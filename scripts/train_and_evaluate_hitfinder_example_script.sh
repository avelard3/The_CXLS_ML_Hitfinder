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

# Load necessary modules
module purge
module load mamba/latest
module load cuda-12.5.0-gcc-12.1.0 

# Activate the conda environment
source activate hitfinder_sol_env_8

# the file path string to the main python script to run the hitfunder model
path_to_script='/your/path/The_CXLS_ML_Hitfinder/src/'
# the main python script name string to run the hitfunder model
script_name='train_and_evaluate_hitfinder.py'

# the file path string to the input lst file containing the file paths to the data to run through the hitfinder model
path_to_input_lst_file='/your/path/'
# the name string of the input lst file containing the file paths to the data to run through the hitfinder model
lst_file_name='your_data_list.lst'

# the string name of the model class to use for the hitfinder model
model_class='CNN_with_Optunas_Best'
# the file path string to the output lst files to save the hitfinder model results
path_to_training_results='/your/path/train_model_output'

# the file path string to the model state dict file to use for the hitfinder model, or None if not using transfer learning
model_class_state_dict=None

# the name string of the model state dict file to use for the hitfinder model
trained_model_state_dict='name_for_new_model.pt'
# the file path string to the output lst files to save the hitfinder model results
path_to_trained_model_state_dict_output='/your/path/hitfinder_models/'

# integer value of the number of epochs to use for the hitfinder model
num_epochs=5
# integer value of the batch size to use for the hitfinder model
batch_size=50
# the string name of the optimizer to use for the hitfinder model
optimizer='Adam'
# the string name of the scheduler to use for the hitfinder model
scheduler='ReduceLROnPlateu'
# the string name of the criterion to use for the hitfinder model
criterion='BCEWithLogitsLoss'
# float value of the learning rate to use for the hitfinder model
learning_rate=0.0005469756482145054


# the string name of the transfer learning model state dict file to use for the hitfinder model, or None if not using transfer learning
transfer_learning=None

# Should the transform defined in prep_loaded_data class Data, be applied
apply_transform=False


#New parameters from optuna
lr_param_patience=51
lr_param_threshold=0.02938279596573081
    
conv_channel_size=1
conv_kernel_size=1
num_linear_dropout_layers=1
linear_layer_size=1
dropout_probability=0.45615248844878264
adam_param_beta1=0.44116899019958417
adam_param_beta2=0.8490932719537443
adam_param_weight_decay=1e-06
batch_norm_2d_momentum=0.33511526193949154
batch_norm_1d_momentum=0.6517130128197814




# creates the transfer learning string if not None
if [ "$transfer_learning" != "None" ]; then
    transfer_learning="${path_to_trained_model_state_dict_output}${model_class_state_dict}"
fi

# Run the Python script with arguments
python ${path_to_script}${script_name} \
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
    -at ${apply_transform} \
    -lrp ${lr_param_patience} \
    -lrt ${lr_param_threshold} \
    -ccs ${conv_channel_size} \
    -cks ${conv_kernel_size} \
    -ldl ${num_linear_dropout_layers} \
    -lls ${linear_layer_size} \
    -dop ${dropout_probability} \
    -ab1 ${adam_param_beta1} \
    -ab2 ${adam_param_beta2} \
    -awd ${adam_param_weight_decay} \
    -bn2dm ${batch_norm_2d_momentum} \
    -bn1dm ${batch_norm_1d_momentum}