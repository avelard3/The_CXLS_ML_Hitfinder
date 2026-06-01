import argparse
from lib import *
import torch
import datetime
from queue import Queue
import numpy as np
import os
from lib import conf

def arguments(parser) -> argparse.ArgumentParser:
    """
    Adds arguments to configure the parameters used for training different models.
    Defined for each input in sbatch script

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
    parser.add_argument('-g', '--geom_file', type=str, help='file path to geometry if multipanel detector, else put None')
    
    parser.add_argument('-hfp', '--hit_file_path_name', type=str, help='Path to hit files if not written into training files')
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
    
    path_to_geom = args.geom_file
    
    transfer_learning_state_dict = args.transfer_learn
    hit_file_path_name = args.hit_file_path_name


    # Transfer learning (yes or no)
    if transfer_learning_state_dict.lower() == 'none':
        transfer_learning_state_dict = None

    cfg = {
        'batch size': batch_size,
        'device': device,
        'epochs': num_epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'learning rate': learning_rate,
        'model': model_arch,
        "lr_param_patience" : conf.lr_param_patience,
        "lr_param_threshold" : conf.lr_param_threshold,
        "adam_param_beta1" : conf.adam_param_beta1,
        "adam_param_beta2" : conf.adam_param_beta2,
        "adam_param_weight_decay" : conf.adam_param_weight_decay,
    }


    executing_mode = 'training'
    path_manager = load_paths.Paths(h5_file_list, executing_mode, path_to_geom) #init Paths object
    path_manager.run_paths() 
    h5_file_paths = path_manager.get_file_names() 
    
    data_manager = load_data.Data(h5_file_paths, executing_mode) #init Data object
    create_data_loader = load_data.CreateDataLoader(data_manager, batch_size) #init CreateDataLoader object that create DataLoader object
    create_data_loader.split_training_data()   
    train_loader, test_loader = create_data_loader.get_training_data_loaders() 
    
    training_manager = train_model.TrainModel(cfg, transfer_learning_state_dict) #init TrainModel object
    training_manager.make_training_instances() 
    training_manager.load_model_state_dict() 
    
    training_manager.assign_new_data(train_loader, test_loader) 
    training_manager.epoch_loop() 
    training_manager.plot_loss_accuracy(training_results) 
        
    # Saving model
    training_manager.save_model(model_dict_save_path) 
    trained_model = training_manager.get_model()
    
    # Checking and reporting accuracy of model
    evaluation_manager = evaluate_model.EvaluateModel(device, trained_model, test_loader) #init EvaluateModel object
    evaluation_manager.run_testing_set() 
    evaluation_manager.make_classification_report()  
    evaluation_manager.plot_confusion_matrix(training_results) 
    evaluation_manager.plot_roc_curve(training_results) 
    
    os.remove("training_vds_delete_me.h5")


if __name__ == '__main__':
    main()