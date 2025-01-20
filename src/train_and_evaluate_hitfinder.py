import argparse
from lib import *
import torch
import datetime
from queue import Queue
import numpy as np

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
    parser.add_argument('-d', '--dict', type=str, help='Output state dict for the traIined model that can be used to load the trained model later.')
    
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
    parser.add_argument('-at', '--apply_transform', type=bool, default = False, help = 'Apply transform to images (true or false)')
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
    
    image_location = args.image_location
    camera_length_location = args.camera_length
    photon_energy_location = args.photon_energy
    peaks_location = args.peaks
    
    transfer_learning_state_dict = args.transfer_learn
    transform = args.apply_transform # Parameter for Data class
    
    
    #temperary holding
    transform = False
    
    master_file = args.master_file
    if master_file == 'None' or master_file == 'none':
        master_file = None
        
        
    # Transfer learning (yes or no)
    # FIXME: There should be a way to get bool to work for this?
    
    if transfer_learning_state_dict == 'None' or transfer_learning_state_dict == 'none':
        transfer_learning_state_dict = None
    
        
    h5_locations = {
        'image': image_location,
        'camera length': camera_length_location,
        'photon energy': photon_energy_location,
        'peak': peaks_location
    }
    
    cfg = {
        'batch size': batch_size,
        'device': device,
        'epochs': num_epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'learning rate': learning_rate,
        'model': model_arch
    }
    
    
    
    executing_mode = 'training'
    path_manager = load_paths.Paths(h5_file_list, h5_locations, executing_mode, master_file)

    path_manager.run_paths()
    
    training_manager = train_model.TrainModel(cfg, h5_locations, transfer_learning_state_dict)
    training_manager.make_training_instances()
    training_manager.load_model_state_dict()

    vds_dataset = path_manager.get_vds()
    h5_file_paths = path_manager.get_file_names()
    
    data_manager = load_data.Data(vds_dataset, h5_file_paths, executing_mode, transform, master_file)
    
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
    evaluation_manager = evaluate_model.ModelEvaluation(cfg, h5_locations, trained_model, test_loader) 
    evaluation_manager.run_testing_set()
    evaluation_manager.make_classification_report()
    evaluation_manager.plot_confusion_matrix(training_results)
    evaluation_manager.plot_roc_curve(training_results)
    

if __name__ == '__main__':
    main()