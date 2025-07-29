import argparse
from lib import *
import torch
import datetime
from queue import Queue
import numpy as np
import os
from lib import conf
from lib import utils as u


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
    parser.add_argument('-o', '--output', type=str, help='Output file path only for training confusion matrix and results.')
    parser.add_argument('-d', '--dict', type=str, help='Output state dict for the trained model that can be used to load the trained model later.')
    
    parser.add_argument('-e', '--epoch', type=int, help='Number of training epochs.')
    parser.add_argument('-b', '--batch', type=int, help='Batch size per epoch for training.')
    parser.add_argument('-op', '--optimizer', type=str, help='Training optimizer function.')
    parser.add_argument('-s', '--scheduler', type=str, help='Training learning rate scheduler.')
    parser.add_argument('-c', '--criterion', type=str, help='Training loss function.')
    parser.add_argument('-lr', '--learning_rate', type=float, help='Training inital learning rate.')
    
    parser.add_argument('-tl', '--transfer_learn', type=str, default=None, help='File path to state dict file for transfer learning.' )
    parser.add_argument('-at', '--apply_transform', type=str, default=False, help='Apply transform to images (true or false)')
  
    
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


def main() -> None:
    """
    This main function is the flow of logic for the training and evaluation of a given model. Here parameter arugments are assigned to variables.
    Classes for data management, training, and evaluation are declared and the relavent functions for the process are called following declaration in blocks. 
    """
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
    training_results = args.output
    model_dict_save_path = args.dict
    
    num_epoch = args.epoch
    batch_size = args.batch
    optimizer = args.optimizer
    scheduler = args.scheduler
    criterion = args.criterion
    learning_rate = args.learning_rate
    
    transfer_learning_state_dict = args.transfer_learn
    transform = args.apply_transform # Parameter for Data class
    
    if transform.lower() == "false":
        transform = False  
    else:
        transform = True


    # Transfer learning (yes or no)
    if transfer_learning_state_dict.lower() == 'none':
        transfer_learning_state_dict = None
        
    
    
    lr_param_patience = conf.lr_param_patience
    lr_param_threshold = conf.lr_param_threshold
    
    conv_channel_size = conf.conv_channel_size
    conv_kernel_size = conf.conv_kernel_size
    num_linear_dropout_layers = conf.num_linear_dropout_layers
    linear_layer_size = conf.linear_layer_size
    dropout_probability = conf.dropout_probability
    adam_param_beta1 = conf.adam_param_beta1
    adam_param_beta2 = conf.adam_param_beta2
    adam_param_weight_decay = conf.adam_param_weight_decay
    batch_norm_2d_momentum = conf.batch_norm_2d_momentum
    batch_norm_1d_momentum = conf.batch_norm_1d_momentum
    
    h5_file_list = args.list
    model_arch = args.model
    model_path = args.dict
    save_output_list = args.output 

    batch_size = args.batch
    
    

    
    cfg = {
        'model': model_arch,
        'model_path': model_path,
        'save_output_list': save_output_list,
        'device': device,
        'batch size': batch_size,
        'epochs': num_epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'learning rate': learning_rate,
        'model': model_arch,
        "lr_param_patience" : lr_param_patience,
        "lr_param_threshold" : lr_param_threshold,
        "adam_param_beta1" : adam_param_beta1,
        "adam_param_beta2" : adam_param_beta2,
        "adam_param_weight_decay" : adam_param_weight_decay,
    }
    
    model_inputs = {
        "conv_channel_size" : conf.conv_channel_size,  
        "conv_kernel_size" : conf.conv_kernel_size,
        "num_linear_dropout_layers" : conf.num_linear_dropout_layers,
        "linear_layer_size" : conf.linear_layer_size,
        "dropout_probability" : conf.dropout_probability,
        "batch_norm_2d_momentum" : conf.batch_norm_2d_momentum,
        "batch_norm_1d_momentum" : conf.batch_norm_1d_momentum
    }

    executing_mode = 'training'
    path_manager = load_paths.Paths(h5_file_list, executing_mode) #init Paths object
    path_manager.run_paths() 
    
    vds_dataset = path_manager.get_vds() 
    h5_file_paths = path_manager.get_file_names() 
    
    data_manager = load_data.Data(vds_dataset, h5_file_paths, executing_mode, transform) #init Data object

    create_data_loader = load_data.CreateDataLoader(data_manager, batch_size) #init CreateDataLoader object that creates DataLoader object
    create_data_loader.run_data_loader() #rename the loader, but single

    data_loader = create_data_loader.get_run_data_loader() 

    # NEED TO GET TRAINED_MODEL SOMEHOW
    # normally train_model.get_model() to get self._model
    
    # Checking and reporting accuracy of model
    
    model = u.LoadModel.make_model_instance(model_arch, model_inputs)
    model = u.LoadModel.load_and_return_model(model_path, model, device)
    evaluation_manager = evaluate_model.EvaluateModel(cfg, model, data_loader) #init EvaluateModel object
    evaluation_manager.run_testing_set() 
    evaluation_manager.make_classification_report()  
    evaluation_manager.plot_confusion_matrix(training_results) 
    evaluation_manager.plot_roc_curve(training_results) 
    
    os.remove("training_vds_delete_me.h5")


if __name__ == '__main__':
    main()