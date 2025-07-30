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
    
    parser.add_argument('-b', '--batch', type=int, help='Batch size per epoch for training.')  
    
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


    
    h5_file_list = args.list
    model_arch = args.model
    model_path = args.dict

    batch_size = args.batch
    
    executing_mode = 'training'
    path_manager = load_paths.Paths(h5_file_list, executing_mode) #init Paths object
    path_manager.run_paths() 
    
    vds_dataset = path_manager.get_vds() 
    h5_file_paths = path_manager.get_file_names() 
    
    data_manager = load_data.Data(h5_file_paths, executing_mode) #init Data object

    create_data_loader = load_data.CreateDataLoader(data_manager, batch_size) #init CreateDataLoader object that creates DataLoader object
    create_data_loader.run_data_loader() #rename the loader, but single

    data_loader = create_data_loader.get_run_data_loader() 

    
    # Checking and reporting accuracy of model
    
    model = u.LoadModel.make_model_instance(model_arch)
    model = u.LoadModel.load_and_return_model(model_path, model, device)
    evaluation_manager = evaluate_model.EvaluateModel(device, model, data_loader) #init EvaluateModel object
    evaluation_manager.run_testing_set() 
    evaluation_manager.make_classification_report()  
    evaluation_manager.plot_confusion_matrix(training_results) 
    evaluation_manager.plot_roc_curve(training_results) 
    
    os.remove("training_vds_delete_me.h5")


if __name__ == '__main__':
    main()