import argparse
from lib import *
import torch
import datetime
from queue import Queue
import numpy as np
import os

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
    
    parser.add_argument('-im', '--image_location', type=str, help='Attribute name for the image')
    parser.add_argument('-cl', '--camera_length', type=str, help='Attribute name for the camera length parameter.')
    parser.add_argument('-pe', '--photon_energy', type=str, help='Attribute name for the photon energy parameter.')
    parser.add_argument('-pk', '--peaks', type=str, help='Attribute name for is there are peaks present.') #aka hit_parameter
    
    parser.add_argument('-tl', '--transfer_learn', type=str, default=None, help='File path to state dict file for transfer learning.' )
    parser.add_argument('-at', '--apply_transform', type=str, default=False, help='Apply transform to images (true or false)')
    
    parser.add_argument('-lrp', '--lr_param_patience', type=int, help='Patience for learning rate parameter input')
    parser.add_argument('-lrt', '--lr_param_threshold', type=float, help='Threshold for learning rate parameter input')
    
    parser.add_argument('-ccs', '--conv_channel_size', type=int, help='Final size of first convolution') #model
    parser.add_argument('-cks', '--conv_kernel_size', type=int, help='Convolution kernel size') #model
    parser.add_argument('-ldl', '--num_linear_dropout_layers', type=int, help='Number of dropout layers and linear layers') #max=3 #model    
    parser.add_argument('-lls', '--linear_layer_size', type=int, help='Size of input of last linear layer')
    parser.add_argument('-dop', '--dropout_probability', type=float, help='Dropout probability') #model
    parser.add_argument('-ab1', '--adam_param_beta1', type=float, help="Adam parameter for momentum")
    parser.add_argument('-ab2', '--adam_param_beta2', type=float, help="Adam parameter for RMSprop")
    parser.add_argument('-awd', '--adam_param_weight_decay', type=float, help="its in the name")
    parser.add_argument('-bn2dm', '--batch_norm_2d_momentum', type=float, help="its in the name")
    parser.add_argument('-bn1dm', '--batch_norm_1d_momentum', type=float, help="its in the name")
  
    
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
        
    
    
    lr_param_patience = args.lr_param_patience
    lr_param_threshold = args.lr_param_threshold
    
    conv_channel_size = args.conv_channel_size
    conv_kernel_size = args.conv_kernel_size
    num_linear_dropout_layers = args.num_linear_dropout_layers
    linear_layer_size = args.linear_layer_size
    dropout_probability = args.dropout_probability
    adam_param_beta1 = args.adam_param_beta1
    adam_param_beta2 = args.adam_param_beta2
    adam_param_weight_decay = args.adam_param_weight_decay
    batch_norm_2d_momentum = args.batch_norm_2d_momentum
    batch_norm_1d_momentum = args.batch_norm_1d_momentum
    
    

    
    cfg = {
        'batch size': batch_size,
        'device': device,
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
        "conv_channel_size" : conv_channel_size,  
        "conv_kernel_size" : conv_kernel_size,
        "num_linear_dropout_layers" : num_linear_dropout_layers,
        "linear_layer_size" : linear_layer_size,
        "dropout_probability" : dropout_probability,
        "batch_norm_2d_momentum" : batch_norm_2d_momentum,
        "batch_norm_1d_momentum" : batch_norm_1d_momentum
    }

    executing_mode = 'training'
    path_manager = load_paths.Paths(h5_file_list, executing_mode)
    
    path_manager.run_paths()
    
    training_manager = train_model.TrainModel(cfg, model_inputs, transfer_learning_state_dict)
    training_manager.make_training_instances()
    training_manager.load_model_state_dict()
    
    vds_dataset = path_manager.get_vds()
    h5_file_paths = path_manager.get_file_names()
    
    data_manager = load_data.Data(vds_dataset, h5_file_paths, executing_mode, transform)
    
    create_data_loader = load_data.CreateDataLoader(data_manager, batch_size)
    
    create_data_loader.split_training_data() 
    train_loader, test_loader = create_data_loader.get_training_data_loaders() 
    
    training_manager.assign_new_data(train_loader, test_loader)
    
    training_manager.epoch_loop() 
    training_manager.plot_loss_accuracy(training_results)
        
    # Saving model
    training_manager.save_model(model_dict_save_path)
    trained_model = training_manager.get_model()
    
    # Checking and reporting accuracy of model
    evaluation_manager = evaluate_model.ModelEvaluation(cfg, trained_model, test_loader) 
    evaluation_manager.run_testing_set()
    evaluation_manager.make_classification_report()
    evaluation_manager.plot_confusion_matrix(training_results)
    evaluation_manager.plot_roc_curve(training_results)
    
    os.remove("training_vds_delete_me.h5")
    
    

if __name__ == '__main__':
    main()