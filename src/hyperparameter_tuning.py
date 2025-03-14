import argparse
from lib import *
import torch
import datetime
from queue import Queue
import numpy as np
import optuna

def arguments(parser) -> argparse.ArgumentParser:
    """
    This function is for adding arguments to configure the parameters used for training different models.
    These parameters are defined the the job sbatch script.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
        
    Returns:
        argparse.ArgumentParser: The parser with the added arugments.
    """
    parser.add_argument('-l', '--list', type=str, help='File path to the .lst file containing file paths to the .h5 file to run through the model.')
    parser.add_argument('-m', '--model', type=str, help='Name of the model architecture class found in models.py that corresponds to the model state dict.')
    
    parser.add_argument('-b', '--batch', type=int, help='Batch size per epoch for training.')
    parser.add_argument('-op', '--optimizer', type=str, help='Training optimizer function.')
    parser.add_argument('-s', '--scheduler', type=str, help='Training learning rate scheduler.')
    parser.add_argument('-c', '--criterion', type=str, help='Training loss function.')
    
    parser.add_argument('-im', '--image_location', type=str, help='Attribute name for the image')
    parser.add_argument('-cl', '--camera_length', type=str, help='Attribute name for the camera length parameter.')
    parser.add_argument('-pe', '--photon_energy', type=str, help='Attribute name for the photon energy parameter.')
    parser.add_argument('-pk', '--peaks', type=str, help='Attribute name for is there are peaks present.') #aka hit_parameter
    
    parser.add_argument('-tl', '--transfer_learn', type=str, default=None, help='File path to state dict file for transfer learning.' )
    parser.add_argument('-at', '--apply_transform', type=bool, default = False, help = 'Apply transform to images (true or false)')
    parser.add_argument('-mf', '--master_file', type=str, default=None, help='File path to the master file containing the .lst files.')
    
    #FIXME: Need to add all the hyperparameters, thre are some missing rn
    parser.add_argument('-er', '--epoch_range', type=int, nargs=2, default=[5,100], help='Lower and upper limit of number of epochs') #optimizer
    parser.add_argument('-lrr', '--learning_rate_range', type=float, nargs=2, default=[0.0001,0.001], help='Lower and upper limit of learning rate') #optimizer
    parser.add_argument('-lrpf', '--lr_param_factor_range', type=float, nargs=2, default=[0.1,0.1], help="")
    parser.add_argument('-lrpp', '--lr_param_patience_range', type=int, nargs=2, default=[3,100], help="")
    parser.add_argument('-lrpt', '--lr_param_threshold_range', type=float, nargs=2, default=[0.1, 0.1], help="")
    
    parser.add_argument('-ccs1', '--conv_channel_size_1_range', type=int, nargs=2, default=[2,8], help='Lower and upper limit of the final size of first convolution') #model
    parser.add_argument('-ccs2', '--conv_channel_size_2_range', type=int, nargs=2, default=[2,8], help='Lower and upper limit of the final size of second convolution') #model
    parser.add_argument('-cks', '--conv_kernel_size_range', type=int, nargs=2, default=[3,3], help='Lower and upper limit of convolution kernel size') #model
    parser.add_argument('-pks', '--pool_kernel_size_range', type=int, nargs=2, default=[2,2], help='Lower and upper limit of pool kernel size') #model
    parser.add_argument('-ldl', '--num_linear_dropout_layers_range', type=int, nargs=2, default=[0,3], help='Lower and upper limit of number of dropout layers and linear layers') #max=3 #model    
    parser.add_argument('-lls1', '--linear_layer_size_1_range', type=int, nargs=2, default=[2,2], help="")
    parser.add_argument('-lls2', '--linear_layer_size_2_range', type=int, nargs=2, default=[2,2], help="")
    parser.add_argument('-dop', '--dropout_probability_range', type=float, nargs=2, default=[0.5,0.8], help='Lower and upper limit of dropout popularity') #model
    # adam parameters (optimizer) 
    
    try:
        args = parser.parse_args()
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}") 
        return args
    
    except argparse.ArgumentError as e:
        print(f"Argument error: {e}")
    
    except argparse.ArgumentTypeError as e:
        print(f"Argument type error: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def objective(trial): #learning rate is a log=true!?

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%m%d%y-%H:%M")
    print(f'Training hitfinder model: {formatted_date_time}')
    
    parser = argparse.ArgumentParser(description='Parameters for training a model.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'This model will be training on: {device}')
    
    # Setting up variables from argument parser
    args = arguments(parser)

    h5_file_list = args.list
    model_arch = args.model
    
    batch_size = args.batch
    optimizer = args.optimizer
    scheduler = args.scheduler
    criterion = args.criterion
    
    image_location = args.image_location
    camera_length_location = args.camera_length
    photon_energy_location = args.photon_energy
    peaks_location = args.peaks
    
    transfer_learning_state_dict = args.transfer_learn
    # FIXME: There should be a way to get bool to work for this?
    if transfer_learning_state_dict == 'None' or transfer_learning_state_dict == 'none':
        transfer_learning_state_dict = None
        
    transform = args.apply_transform # Parameter for Data class
    transform = False  #temperary holding
    
    master_file = args.master_file
    if master_file == 'None' or master_file == 'none':
        master_file = None
        
    # hyperparameter tuning #? maybe add them all to a dictionary here
    epoch_range = tuple(args.epoch_range)
    learning_rate_range = tuple(args.learning_rate_range)
    lr_param_factor_range = tuple(args.lr_param_factor_range)
    lr_param_patience_range = tuple(args.lr_param_patience_range)
    lr_param_threshold_range = tuple(args.lr_param_threshold_range)
    
    conv_channel_size_1_range = tuple(args.conv_channel_size_1_range)    
    conv_channel_size_2_range = tuple(args.conv_channel_size_2_range)
    conv_kernel_size_range = tuple(args.conv_kernel_size_range)
    pool_kernel_size_range = tuple(args.pool_kernel_size_range)
    num_linear_dropout_layers_range = tuple(args.num_linear_dropout_layers_range)
    linear_layer_size_1_range = tuple(args.linear_layer_size_1_range)
    linear_layer_size_2_range = tuple(args.linear_layer_size_2_range)
    dropout_probability_range = tuple(args.dropout_probability_range)


    h5_locations = {
        'image': image_location,
        'camera length': camera_length_location,
        'photon energy': photon_energy_location,
        'peak': peaks_location
    }
    
    cfg = {
        'batch size': batch_size,
        'device': device,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'model': model_arch
    }

    
    # needed in train_model.py
    epoch = trial.suggest_int('epoch', epoch_range[0], epoch_range[1]) #not model
    learning_rate = trial.suggest_float('learning_rate', learning_rate_range[0], learning_rate_range[1], log=True) #not model
    lr_param_factor = trial.suggest_float('lr_param_factor', lr_param_factor_range[0], lr_param_factor_range[1]) #not model
    lr_param_patience = trial.suggest_int('lr_param_patience', lr_param_patience_range[0], lr_param_patience_range[1]) #not model
    lr_param_threshold = trial.suggest_float('lr_param_threshold', lr_param_threshold_range[0], lr_param_threshold_range[1]) #not model
    
    # needed in models.py
    conv_channel_size_1 = trial.suggest_int('conv_channel_size_1', conv_channel_size_1_range[0], conv_channel_size_1_range[1]) 
    conv_channel_size_2 = trial.suggest_int('conv_channel_size_2', conv_channel_size_2_range[0], conv_channel_size_2_range[1])
    conv_kernel_size = trial.suggest_int('conv_kernel_size', conv_kernel_size_range[0], conv_kernel_size_range[1]) #?
    pool_kernel_size = trial.suggest_int('pool_kernel_size', pool_kernel_size_range[0], pool_kernel_size_range[1])
    num_linear_dropout_layers = trial.suggest_int('num_linear_dropout_layers', num_linear_dropout_layers_range[0], num_linear_dropout_layers_range[1])
    linear_layer_size_1 = trial.suggest_int('linear_layer_size_1', linear_layer_size_1_range[0], linear_layer_size_1_range[1])
    linear_layer_size_2 = trial.suggest_int('linear_layer_size_2', linear_layer_size_2_range[0], linear_layer_size_2_range[1])
    dropout_probability = trial.suggest_float('dropout_probability', dropout_probability_range[0], dropout_probability_range[1]) 
    
    hyperparam_dict_train = {
        "epoch" : epoch,
        "learning_rate" : learning_rate,
        "lr_param_factor" : lr_param_factor,
        "lr_param_patience" : lr_param_patience,
        "lr_param_threshold" : lr_param_threshold
        
        
    }
    hyperparam_dict_model = {
        "conv_channel_size_1" : conv_channel_size_1,  
        "conv_channel_size_2" : conv_channel_size_2,
        "conv_kernel_size" : conv_kernel_size,
        "pool_kernel_size" : pool_kernel_size,
        "num_linear_dropout_layers" : num_linear_dropout_layers,
        "linear_layer_size_1" : linear_layer_size_1,
        "linear_layer_size_2" : linear_layer_size_2,
        "dropout_probability" : dropout_probability
    }
    
    executing_mode = 'training'
    path_manager = load_paths.Paths(h5_file_list, h5_locations, executing_mode, master_file)

    path_manager.run_paths()
    
    #so i think if i change cfg those would be the inputs that need to be changed
    tuning_manager = tune_model.TuneModel(cfg, hyperparam_dict_train, hyperparam_dict_model, h5_locations, transfer_learning_state_dict)
    tuning_manager.make_training_instances()
    tuning_manager.load_model_state_dict()

    vds_dataset = path_manager.get_vds()
    h5_file_paths = path_manager.get_file_names()
    
    data_manager = load_data.Data(vds_dataset, h5_file_paths, executing_mode, transform, master_file)
    create_data_loader = load_data.CreateDataLoader(data_manager, batch_size)
    create_data_loader.split_training_data() 
    train_loader, test_loader = create_data_loader.get_training_data_loaders() 
    
    tuning_manager.assign_new_data(train_loader, test_loader)
    tuning_manager.epoch_loop()
    

    #! add adam parameters
    #consolidate


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print("Best Hyperparameters:", study.best_params)