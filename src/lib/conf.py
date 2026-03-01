# This file is for holding values used in the hitfinder code

# Size of image used (must be same for training and running), must be a square
required_image_size = (512 , 512) 

# Places where data could be stored in h5 files (especially when you have a mix of files from different sources)
possible_image_paths = ['/images/', 'entry/data/data', '/entry_1/data_1/data']
possible_camera_length_paths = ['/detector_distance', 'entry/instrument/detector/detector_distance', 'LCLS/detector_1/EncoderValue']
possible_photon_energy_paths = ['/photon_energy_eV', 'entry/instrument/beam/incident_wavelength', 'LCLS/photon_energy_eV']
possible_hit_parameter_paths = ['/hit','/hits/hits','/hits', '/hit/', '/hits/', 'hits']

# Hyperparameters that were optimized by Optuna (stored as conf variables for ease of hyperparameter testing)
lr_param_patience=10
lr_param_threshold=1e-3   
conv_channel_size=2
conv_kernel_size=3
num_linear_dropout_layers=1
linear_layer_size=2
dropout_probability=0.3
adam_param_beta1=0.9
adam_param_beta2=0.999
adam_param_weight_decay=1e-4
batch_norm_2d_momentum=0.1
batch_norm_1d_momentum=0.1