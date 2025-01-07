import argparse
from lib import *
import torch
import datetime
from queue import Queue


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
    parser.add_argument('-mf', '--master_file', type=str, default=None, help='File path to the master file containing the .lst files.')
    
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
    
    image_location = args.image_location
    camera_length_location = args.camera_length
    photon_energy_location = args.photon_energy
    peaks_location = None
    batch_size = args.batch
    
    # Temperary hold
    transform = False
    
    master_file = args.master_file
    if master_file == 'None' or master_file == 'none':
        master_file = None
    
    h5_locations = {
        'image': image_location,
        'camera length': camera_length_location,
        'photon energy': photon_energy_location,
        'peak': peaks_location
    }
    
    cfg = {
        'model': model_arch,
        'model_path': model_path,
        'save_output_list': save_output_list,
        'device': device,
    }
    executing_mode = 'running'
    path_manager = load_paths.Paths(h5_file_list, h5_locations, executing_mode, master_file)

    path_manager.run_paths()
    
    
    process_data = run_model.RunModel(cfg, h5_locations)
    process_data.make_model_instance()
    process_data.load_model()

    vds_dataset = path_manager.get_vds()    
    h5_file_paths = path_manager.get_file_names()

    data_manager = load_data.Data(vds_dataset, h5_file_paths, executing_mode, transform, master_file)

    create_data_loader = load_data.CreateDataLoader(data_manager, batch_size)
    create_data_loader.inference_data_loader()

    data_loader = create_data_loader.get_inference_data_loader()
    print(f"data_loader shape from run_hitfinder_model {data_loader}")

    process_data.classify_data(data_loader) 
   
    process_data.create_model_output_lst_files()

if __name__ == '__main__':
    main()