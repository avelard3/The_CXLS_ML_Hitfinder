import argparse
from lib import *
import torch
import datetime
from queue import Queue

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
    
    parser.add_argument('-cl', '--camera_length', type=str, help='Attribute name for the camera length parameter.')
    parser.add_argument('-pe', '--photon_energy', type=str, help='Attribute name for the photon energy parameter.')
    parser.add_argument('-pk', '--peaks', type=str, help='Attribute name for is there are peaks present.')
    
    parser.add_argument('-tl', '--transfer_learn', type=str, default=None, help='File path to state dict file for transfer learning.' )
    parser.add_argument('-at', '--apply_transform', type=bool, default = False, help = 'Apply transform to images (true or false)')
    parser.add_argument('-me', '--multievent', type=str, help='True or false value for if the input .h5 files are multievent or not.')
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
    
    camera_length = args.camera_length
    photon_energy = args.photon_energy
    peaks = args.peaks
    
    transfer_learning_state_dict = args.transfer_learn
    transform = args.apply_transform # Parameter for Data class
    multievent = args.multievent
    master_file = args.master_file
    
    #temperary holding
    
    if master_file == 'None' or master_file == 'none':
        master_file = None
        
        
    # Transfer learning (yes or no)
    # FIXME: There should be a way to get bool to work for this?
    
    if transfer_learning_state_dict == 'None' or transfer_learning_state_dict == 'none':
        transfer_learning_state_dict = None
    
    
    print("train and evaluate .py line 102", transform)
    
    attributes = {
        'camera length': camera_length,
        'photon energy': photon_energy,
        'peak': peaks
    }
    
    # Create a queue for files to keep from overwhelming the computer (so you can just iterate one by one)
    if multievent == 'True' or multievent == 'true':
        path_manager = load_paths.PathsMultiEvent(h5_file_list,  attributes,  master_file)
    else:
        path_manager = load_paths.PathsSingleEvent(h5_file_list, attributes, master_file)
        
    path_manager.read_file_paths()
    h5_file_path_queue = path_manager.get_file_path_queue()
    
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
    
    training_manager = train_model.TrainModel(cfg, attributes, transfer_learning_state_dict)
    training_manager.make_training_instances()
    training_manager.load_model_state_dict()

    # Iterate through h5 files to split data and do manager/loader
    # FIXME: I don't understand anything

    while not h5_file_path_queue.empty():
        path_manager.process_files()

        h5_tensor_list = path_manager.get_h5_tensor_list()
        h5_attribute_list = path_manager.get_h5_attribute_list()
        h5_file_paths = path_manager.get_h5_file_paths()
        data_manager = load_data.Data(h5_tensor_list, h5_attribute_list, h5_file_paths, transform)
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
    evaluation_manager = evaluate_model.ModelEvaluation(cfg, attributes, trained_model, test_loader) #error here said that test_loader doesn't have a value i think
    evaluation_manager.run_testing_set()
    evaluation_manager.make_classification_report()
    evaluation_manager.plot_confusion_matrix(training_results)
    
    
    
if __name__ == '__main__':
    main()