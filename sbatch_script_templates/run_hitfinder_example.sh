#!/bin/bash

#SBATCH -N 1
#SBATCH -c 12
#SBATCH -G h100:1
#SBATCH -t 0-00:60:00
#SBATCH --mem=128G
#SBATCH -p general
#SBATCH -q grp_cxfel 
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
conda activate hitfinder_sol_env
export HDF5_PLUGIN_PATH=/home/avelard3/.local/lib/python3.12/site-packages/hdf5plugin/plugins


# the file path string to the main python script to run the hitfunder model
path_to_script='/scratch/avelard3/The_CXLS_ML_Hitfinder/src/'
# the main python script name string to run the hitfunder model
script_name='run_hitfinder_model.py'

# the file path string to the input lst file containing the file paths to the data to run through the hitfinder model
path_to_input_lst_file='/scratch/avelard3/'
# the name string of the input lst file containing the file paths to the data to run through the hitfinder model
input_lst_name='specific_master_for_train.lst'

# the string name of the model class to use for the hitfinder model
model_class='CNN_with_Optunas_Best'
# the file path string to the model state dict file to use for the hitfinder model
path_to_model_state_dict='/home/avelard3/hitfinder_models/'
# the name string of the model state dict file to use for the hitfinder model
model_state_dict='CNN_with_Optunas_Master_dec17.pt'

# the file path string to the output lst files to save the hitfinder model results
path_to_output_lst_files='/home/avelard3/hitfinder_output_files/run_model_output'

# file path to geometry if multipanel detector, else put None #!!!!!!!!!!!!!!!!!!
path_to_geom=None

# integer value of the batch size to use for the hitfinder model
batch_size=10 # Larger batch size is faster but more memory

# Run the Python script with arguments
python ${path_to_script}${script_name} \
    -l ${path_to_input_lst_file}${input_lst_name} \
    -m ${model_class} \
    -d ${path_to_model_state_dict}${model_state_dict} \
    -o ${path_to_output_lst_files} \
    -b ${batch_size} \
    -g ${path_to_geom}

