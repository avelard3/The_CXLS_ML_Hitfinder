#import os
import argparse
from lib import *
import torch
import datetime

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import numpy as np

CLASSES = 10


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
    
    #Eventually I should make all the hyperparameters inputs too, but not right now
    
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



def define_things(self) -> None:
    """
    This main function is the flow of logic for the training and evaluation of a given model. Here parameter arugments are assigned to variables.
    Classes for data management, training, and evaluation are declared and the relavent functions for the process are called following declaration in blocks. 
    """
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%m%d%y-%H:%M")
    print(f'Training hitfinder model: {formatted_date_time}')
    
    parser = argparse.ArgumentParser(description='Parameters for training a model.')
    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #! Where is device defined/gotten when training or running??
    #It's passed into the TrainModel with cfg.... do I need to change that too then???
    print(f'This model will be training on: {device}')
    
    # Setting up variables from argument parser
    args = arguments(parser)
    self._h5_file_list = args.list
    model_arch = args.model
    self._training_results = args.output
    self._model_dict_save_path = args.dict
    
    self._num_epoch = args.epoch
    self._batch_size = args.batch
    optimizer = args.optimizer
    scheduler = args.scheduler
    criterion = args.criterion
    learning_rate = args.learning_rate
    
    image_location = args.image_location
    camera_length_location = args.camera_length
    photon_energy_location = args.photon_energy
    peaks_location = args.peaks
    
    self._transfer_learning_state_dict = args.transfer_learn
    self._transform = args.apply_transform # Parameter for Data class
    
    self._N_TRAIN_EXAMPLES = self._batch_size * 30
    self._N_VALID_EXAMPLES = self._batch_size * 10
    #temperary holding
    self._transform = False
    
    self._master_file = args.master_file
    if self._master_file == 'None' or self._master_file == 'none':
        self._master_file = None
        
        
    # Transfer learning (yes or no)
    # FIXME: There should be a way to get bool to work for this?
    
    if self._transfer_learning_state_dict == 'None' or self._transfer_learning_state_dict == 'none':
        self._transfer_learning_state_dict = None
    
        
    self._h5_locations = {
        'image': image_location,
        'camera length': camera_length_location,
        'photon energy': photon_energy_location,
        'peak': peaks_location
    }
    
    self._cfg = {
        'batch size': self._batch_size,
        'device': self._device,
        'epochs': self._num_epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'learning rate': learning_rate,
        'model': model_arch
    }
    
    # Generate the model.
def objective(self, trial):
    
    self.define_things()
    
    model = define_model(trial).to(self._device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]) 
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # convolution kernel size
    # convolution channels
    # dropout probability
    # learning rate
    # epochs
    # adam parameters
    # learning rate scheduler parameters
    
    executing_mode = 'training' 
    path_manager = load_paths.Paths(self._h5_file_list, self._h5_locations, executing_mode, self._master_file) #good?

    path_manager.run_paths() #good?
    
    training_manager = train_model.TrainModel(self._cfg, self._h5_locations, self._transfer_learning_state_dict)
    training_manager.make_training_instances()
    training_manager.load_model_state_dict()

    vds_dataset = path_manager.get_vds()
    h5_file_paths = path_manager.get_file_names()
    
    data_manager = load_data.Data(vds_dataset, h5_file_paths, executing_mode, self._transform, self._master_file)
    
    create_data_loader = load_data.CreateDataLoader(data_manager, self._batch_size)

    create_data_loader.split_training_data() 
    train_loader, test_loader = create_data_loader.get_training_data_loaders() #* #! I made the assumption that test_loader and valid_loader were the same thing
    
    
    #where model is called?
    for epoch in range(self.from django.utils.translation import ugettext_lazy as _num_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * self._batch_size >= self._N_TRAIN_EXAMPLES: #I am very unsure what N_TRAIN_EXAMPLES and N_VALID_EXAMPLES are so I'm leaving those for now
                break

            data, target = data.view(data.size(0), -1).to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # Limiting validation data.
                if batch_idx * self._batch_size >= self._N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(test_loader.dataset), self._N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return accuracy
    
    
    
    
    
    ##################################!
                
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


def define_model(self, trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    # Assuming that input_channels & output_channels = 1 and input_size = (512, 512)
    input_channels = 1 # should this include the input_size????
    output_channels = 1
    for i in range(n_layers):
        #*Things that should vary inside this loop*#
        self.kernel_size = 0
        self.stride = 0
        self.pool_kernel_size = 0
        
        #**#
        
        ouput_channels = trial.suggest_int("n_units_l{}".format(i), 4, 128) #the 4,8,16 variable.... num_conv_channels?
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)) # layers.append(nn.Linear(in_features, out_features))
        layers.append(F.relu()) #does this need an input?? like relu(x).... is that maybe why it was nn.ReLU or whtever?
        
        # p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5) #* dropout!!!
        # layers.append(nn.Dropout(p))

        input_channels = output_channels
    layers.append(nn.Linear(input_channels, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

    self.kernel_size1 = 3
     self.stride1 = 1
    self.padding1 = 1
    self.kernel_size2 = 3
    self.stride2 = 1
    self.padding2 = 1
#below here should just be one time and it can decide the optimal numbwer of layers inside the for loop
    # self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(4, 8, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
    self.pool2 = nn.MaxPool2d(2,2)
    self.conv3 = nn.Conv2d(8, 16, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
    self.pool3 = nn.MaxPool2d(2,2)
    out_height1 = self.calculate_output_dimension(input_size[0], self.kernel_size1, self.stride1, self.padding1)
    out_width1 = self.calculate_output_dimension(input_size[1], self.kernel_size1, self.stride1, self.padding1)
    out_height2 = self.calculate_output_dimension(out_height1 // 2, self.kernel_size2, self.stride2, self.padding2)
    out_width2 = self.calculate_output_dimension(out_width1 // 2, self.kernel_size2, self.stride2, self.padding2)
    self.fc_size_1 = 0.25 * out_height2 * out_width2 #fully connected layer
    self.fc1 = nn.Linear(self.fc_size_1, output_channels)
    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1
    def forward(self, x, camera_length, photon_energy):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        #*this should be outside the for loop
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        return x




if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600) #objective is the function that originally created everything and ran the epochs and torch.no_grad()

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))