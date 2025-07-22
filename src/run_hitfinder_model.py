import argparse
from lib import *
import torch
import datetime
from queue import Queue
import os


def arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
    """
    This function is for adding an argument when running the python file. 
    It needs to take an lst file of the h5 files for the model use. 
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
        
    Returns:
        argparse.ArgumentParser: The parser with the added arugments.
    """
    parser.add_argument('-l', '--list', type=str, help='File path to the .lst file containing file paths to the .h5 file to run through the model.')
    parser.add_argument('-m', '--model', type=str, help='Name of the model architecture class found in models.py that corresponds to the model state dict.')
    parser.add_argument('-d', '--dict', type=str, help='File path to the model state dict .pt file.')
    parser.add_argument('-o', '--output', type=str, help='Output file path only for the .lst files after classification.')
    
    parser.add_argument('-im', '--image_location', type=str, help='Attribute name for the image')
    parser.add_argument('-cl', '--camera_length', type=str, help='Attribute name for the camera length parameter.')
    parser.add_argument('-pe', '--photon_energy', type=str, help='Attribute name for the photon energy parameter.')
    parser.add_argument('-b', '--batch', type=int, help='Batch size for data running through the model.')
    
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


def main():
    """
    This main function is the flow of logic for running a trained model. Here parameter arugments are assigned to variables.
    Classes for data management and using the model are declared and the relavent functions for the process are called following declaration in blocks. 
    """
    #tic
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%m%d%y-%H:%M")
    print(f'Starting hitfinder model: {formatted_date_time}')
    
    parser = argparse.ArgumentParser(description='Parameters for running a model.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'This model will be running on: {device}')

    args = arguments(parser)
    h5_file_list = args.list
    model_arch = args.model
    model_path = args.dict
    save_output_list = args.output 

    peaks_location = None
    batch_size = args.batch
    
    # Temperary hold
    transform = False # there's no reason for transform to be true for running; the only thing that might be done is 
    
    cfg = {
        'model': model_arch,
        'model_path': model_path,
        'save_output_list': save_output_list,
        'device': device,
    }
    
    #! DELETE ME LATER
    conv_channel_size=7
    conv_kernel_size=6
    num_linear_dropout_layers=1
    linear_layer_size=6
    dropout_probability=0.38437557450917537
    batch_norm_2d_momentum=0.17464996633777477
    batch_norm_1d_momentum=0.7345315119311252




    model_inputs = {
        "conv_channel_size" : conv_channel_size,  
        "conv_kernel_size" : conv_kernel_size,
        "num_linear_dropout_layers" : num_linear_dropout_layers,
        "linear_layer_size" : linear_layer_size,
        "dropout_probability" : dropout_probability,
        "batch_norm_2d_momentum" : batch_norm_2d_momentum,
        "batch_norm_1d_momentum" : batch_norm_1d_momentum
    }

    #! DELETE ME LATER END except that i had to change things other places so i might just be stuck like this now
    
    
    executing_mode = 'running'
    path_manager = load_paths.Paths(h5_file_list, executing_mode)

    path_manager.run_paths()
    
    
    process_data = run_model.RunModel(cfg, model_inputs)
    process_data.make_model_instance()
    process_data.load_model()

    vds_dataset = path_manager.get_vds()    
    h5_file_paths = path_manager.get_file_names()

    data_manager = load_data.Data(vds_dataset, h5_file_paths, executing_mode, transform)

    create_data_loader = load_data.CreateDataLoader(data_manager, batch_size)
    create_data_loader.inference_data_loader()

    data_loader = create_data_loader.get_inference_data_loader()

    process_data.classify_data(data_loader) 
   
    process_data.create_model_output_lst_files()
    
    os.remove("running_vds_delete_me.h5")

if __name__ == '__main__':
    main()