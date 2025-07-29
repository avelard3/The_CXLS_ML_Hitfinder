import argparse
from lib import *
import torch
import datetime
from queue import Queue
import numpy as np
import optuna
import plotly
import logging
import sys
import pickle

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
    
    #FIXME: Need to add all the hyperparameters, thre are some missing rn
    parser.add_argument('-er', '--epoch_range', type=int, nargs=2, default=[5,100], help='Lower and upper limit of number of epochs') #optimizer
    parser.add_argument('-lrr', '--learning_rate_range', type=float, nargs=2, default=[0.0001,0.001], help='Lower and upper limit of learning rate') #optimizer
    parser.add_argument('-lrpp', '--lr_param_patience_range', type=int, nargs=2, default=[3,100], help="")
    parser.add_argument('-lrpt', '--lr_param_threshold_range', type=float, nargs=2, default=[0.1, 0.1], help="")
    
    parser.add_argument('-ccs', '--conv_channel_size_range', type=int, nargs=2, default=[2,8], help='Lower and upper limit of the final size of first convolution') #model
    parser.add_argument('-cks', '--conv_kernel_size_range', type=int, nargs=2, default=[3,3], help='Lower and upper limit of convolution kernel size') #model
    parser.add_argument('-ldl', '--num_linear_dropout_layers_range', type=int, nargs=2, default=[1,3], help='Lower and upper limit of number of dropout layers and linear layers') #max=3 #model    
    parser.add_argument('-lls', '--linear_layer_size_range', type=int, nargs=2, default=[2,2], help="")
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
    if transfer_learning_state_dict.lower() == 'none':
        transfer_learning_state_dict = None

        
    # hyperparameter tuning #? maybe add them all to a dictionary here
    epoch_range = tuple(args.epoch_range)
    learning_rate_range = tuple(args.learning_rate_range)
    lr_param_patience_range = tuple(args.lr_param_patience_range)
    lr_param_threshold_range = tuple(args.lr_param_threshold_range)
    
    conv_channel_size_range = tuple(args.conv_channel_size_range)    
    conv_kernel_size_range = tuple(args.conv_kernel_size_range)
    num_linear_dropout_layers_range = tuple(args.num_linear_dropout_layers_range)
    linear_layer_size_range = tuple(args.linear_layer_size_range)
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
    lr_param_patience = trial.suggest_int('lr_param_patience', lr_param_patience_range[0], lr_param_patience_range[1]) #not model
    lr_param_threshold = trial.suggest_float('lr_param_threshold', lr_param_threshold_range[0], lr_param_threshold_range[1]) #not model
    
    # needed in models.py
    conv_channel_size = trial.suggest_int('conv_channel_size', conv_channel_size_range[0], conv_channel_size_range[1]) 
    conv_kernel_size = trial.suggest_int('conv_kernel_size', conv_kernel_size_range[0], conv_kernel_size_range[1]) #?
    num_linear_dropout_layers = trial.suggest_int('num_linear_dropout_layers', num_linear_dropout_layers_range[0], num_linear_dropout_layers_range[1])
    linear_layer_size = trial.suggest_int('linear_layer_size', linear_layer_size_range[0], linear_layer_size_range[1])
    dropout_probability = trial.suggest_float('dropout_probability', dropout_probability_range[0], dropout_probability_range[1]) 
    beta1 = trial.suggest_float('beta1', 0.1000, 1.0000)
    beta2 = trial.suggest_float('beta2', 0.1000, 1.0000)
    weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    momentum_2d = trial.suggest_float('momentum_2d', 0.001, 0.99)
    momentum_1d = trial.suggest_float('momentum_1d', 0.001, 0.99)

     #FIXME Tehcnically this should be in the sbatch script but I'm just trying to get it done at this point and i dont anticipate it needing to be any different because i'm researching this so much
    
    
    
    hyperparam_dict_train = {
        "epoch" : epoch,
        "learning_rate" : learning_rate,
        "lr_param_patience" : lr_param_patience,
        "lr_param_threshold" : lr_param_threshold,
        "beta1" : beta1,
        "beta2" : beta2,
        "weight_decay" : weight_decay,
    }
    
    print(hyperparam_dict_train)
    
    hyperparam_dict_model = {
        "conv_channel_size" : conv_channel_size,  
        "conv_kernel_size" : conv_kernel_size,
        "num_linear_dropout_layers" : num_linear_dropout_layers,
        "linear_layer_size" : linear_layer_size,
        "dropout_probability" : dropout_probability,
        "momentum_2d" : momentum_2d,
        "momentum_1d" : momentum_1d
    }
    
    executing_mode = 'training'
    print("Creating Paths object")
    path_manager = load_paths.Paths(h5_file_list, h5_locations, executing_mode)
    print("Running load paths")
    path_manager.run_paths()
    print("Creating TuneModel object")
    tuning_manager = tune_model.TuneModel(cfg, hyperparam_dict_train, hyperparam_dict_model, h5_locations, transfer_learning_state_dict)
    print("Create training instance")
    tuning_manager.make_training_instances() 
    print("Loading model state dictionary")
    tuning_manager.load_model_state_dict()
    print("Getting VDS")
    vds_dataset = path_manager.get_vds()
    print("Get file names")
    h5_file_paths = path_manager.get_file_names()
    print("Creating Data object")
    data_manager = load_data.Data(vds_dataset, h5_file_paths, executing_mode)
    print("Creating DataLoader")
    create_data_loader = load_data.CreateDataLoader(data_manager, batch_size)
    print("Split training data")
    create_data_loader.split_training_data() 
    print("Getting train and test loader")
    train_loader, test_loader = create_data_loader.get_training_data_loaders() 
    print("Assigning new data")
    tuning_manager.assign_new_data(train_loader, test_loader)
    print("Epoch loop starting")
    train_loss_from_epoch = tuning_manager.epoch_loop(trial)
    print("After epoch loop the train loss is", train_loss_from_epoch)
    return train_loss_from_epoch
    #consolidate

def create_optimization_history_plot(study, path:str=None) -> None:
        try:
            print('Creating optimization history plot...')
            optimization_history = optuna.visualization.plot_optimization_history(study)
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'optimization_history.png'
                optimization_history.write_image(path)
                
                print(f'Optimization history plot saved to: {path}')
        except Exception as e:
            print(f"An error occurred while creating the optimization history plot: {e}")  
            
            
def create_intermediate_values_plot(study, path:str=None) -> None:
        try:
            print('Creating intermediate values plot...')
            intermediate_values = optuna.visualization.plot_intermediate_values(study)
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'intermediate_values.png'
                intermediate_values.write_image(path)
                
                print(f'Intermediate values plot saved to: {path}')
        except Exception as e:
            print(f"An error occurred while creating the intermediate values plot: {e}")  
            

def create_slice_plot(study, path:str=None) -> None:
        try:
            print('Creating slice plot...')
            slice = optuna.visualization.plot_slice(study)
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'slice.png'
                slice.write_image(path)
                
                print(f'Slice plot saved to: {path}')
        except Exception as e:
            print(f"An error occurred while creating the slice plot: {e}")  

def create_param_importances_plot(study, path:str=None) -> None:
        try:
            print('Creating param importances plot...')
            param_importances = optuna.visualization.plot_param_importances(study)
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'param_importances.png'
                param_importances.write_image(path)
                
                print(f'Param importances plot saved to: {path}')
        except Exception as e:
            print(f"An error occurred while creating the param importances plot: {e}")  
            
def create_edf_plot(study, path:str=None) -> None:
        try:
            print('Creating edf plot...')
            edf = optuna.visualization.plot_edf(study)
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'edf.png'
                edf.write_image(path)
                
                print(f'Edf plot saved to: {path}')
        except Exception as e:
            print(f"An error occurred while creating the edf plot: {e}")  
            
def create_timeline_plot(study, path:str=None) -> None:
        try:
            print('Creating timeline plot...')
            timeline = optuna.visualization.plot_timeline(study)
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'timeline.png'
                timeline.write_image(path)
                
                print(f'Timeline plot saved to: {path}')
        except Exception as e:
            print(f"An error occurred while creating the timeline plot: {e}")  




if __name__ == '__main__':
    study_name = "the-cxls-ml-hitfinder-trial1_june23"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)

    study.optimize(objective, n_trials=15) #!

    
    image_path_save = '/scratch/avelard3/big_files/pics_from_optuna_5_9'
    create_optimization_history_plot(study, image_path_save)
    create_intermediate_values_plot(study, image_path_save)
    create_slice_plot(study, image_path_save)
    create_param_importances_plot(study, image_path_save)
    create_edf_plot(study, image_path_save)
    create_timeline_plot(study, image_path_save)
    
    
    print("Best Hyperparameters:", study.best_params)
