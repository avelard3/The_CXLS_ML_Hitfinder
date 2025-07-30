import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import datetime
from torch.utils.data import DataLoader

from . import models as m
from . import conf
from . import utils as u


class TrainModel:
    
    def __init__(self, cfg: dict, model_inputs: dict, transfer_learning_state_dict: str) -> None:
        """
        This constructor breaks up the training configuration infomation dictionary and h5 metadata key dictionary.
        In addition, a logging object is created and global list are created for storing infomation about the training loss and accuracy. 

        Args:
            cfg (dict): Dictionary containing important information for training including: data loaders, batch size, training device, number of epochs, the optimizer, the scheduler, the criterion, the learning rate, and the model class. Everything besides the data loaders and device are arguments in the sbatch script.
            attributes (dict): Dictionary containing the names of the metadata contained in the h5 image files. These names could change depending on whom created the metadata, so the specific names are arguments in the sbatch script. 
            transfer_learning_state_dict (str): String to location of transfer learning state dictionary
        """

        self._transfer_learning_path = transfer_learning_state_dict

        self._train_loader = None
        self._test_loader = None
        self._device = cfg['device']
        self._epochs = cfg['epochs']
        self._optimizer = cfg['optimizer']
        self._scheduler = cfg['scheduler']
        self._criterion = cfg['criterion']
        self._learning_rate = cfg['learning rate']
        self._model = cfg['model']        
        
        self._lr_param_patience = cfg["lr_param_patience"]
        self._lr_param_threshold = cfg["lr_param_threshold"]
        self._adam_param_beta1 = cfg["adam_param_beta1"]
        self._adam_param_beta2 = cfg["adam_param_beta2"]
        self._adam_param_weight_decay = cfg["adam_param_weight_decay"]
        
        self._model_in = model_inputs

        
        self._plot_train_accuracy = np.zeros(self._epochs)
        self._plot_train_loss = np.zeros(self._epochs)
        self._plot_test_accuracy = np.zeros(self._epochs)
        self._plot_test_loss = np.zeros(self._epochs)
        
        
        
    def assign_new_data(self, train: DataLoader, test: DataLoader) -> None:
        """
        Assigns new data loaders. 
        
        Args:
            train (DataLoader): DataLoader for loading data for training only
            test (DataLoader): DataLoader for loading data for testing only
        """
        self._train_loader = train
        self._test_loader = test

        
    def make_training_instances(self) -> None:
        """
        This function takes the strings from the sbatch script and makes them objects.
        These strings are objects that are needed for the training. Objects declared here are :
        - the model
        - the optimizer
        - the learning rate scheduler
        - the loss criterion
        
        Raises:
            AttributeError: If variable name is not found in torch
            TypeError: If variable is not callable
            Exception: other
        """
        try:
            self._model = getattr(m, self._model)(model_inputs=self._model_in).to(self._device)
            self._optimizer = getattr(optim, self._optimizer)(self._model.parameters(), lr=self._learning_rate, betas=[self._adam_param_beta1, self._adam_param_beta2], weight_decay=self._adam_param_weight_decay)            
            self._scheduler = getattr(lrs, self._scheduler)(self._optimizer, mode='min', factor=0.1, patience=self._lr_param_patience, threshold=self._lr_param_threshold) # learning rate scheduler #probably specific to optimizer
            self._criterion = getattr(nn, self._criterion)() # loss function. should probably leave that alone for now
            
            print('All training objects have been created.')
            
        except AttributeError as e:
            print(f"AttributeError: {e}")
            if 'nn' in str(e):
                print(f"Error: '{self._model_name}', '{self._criterion_name}' not found in torch.nn")
            elif 'optim' in str(e):
                print(f"Error: '{self._optimizer_name}' not found in torch.optim")
            elif 'lr_scheduler' in str(e):
                print(f"Error: '{self._scheduler_name}' not found in torch.optim.lr_scheduler")
        except TypeError as e:
            print(f"TypeError: {e}")
            if 'nn' in str(e):
                print(f"Error: '{self._model_name}' or '{self._criterion_name}' is not callable")
            elif 'optim' in str(e):
                print(f"Error: '{self._optimizer_name}' is not callable")
            elif 'lr_scheduler' in str(e):
                print(f"Error: '{self._scheduler_name}' is not callable")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def load_model_state_dict(self) -> None:
        """
        This function loads in the state dict of a model if provided.
        """
        if self._transfer_learning_path != None:
            print("Reading transfer learning", self._transfer_learning_path)
            self._model = u.LoadModel.load_and_return_model(self._transfer_learning_path, self._model, self._device)

        else:
            print(f'There is no model state dict to load into: {self._model.__class__.__name__}')
    
    def epoch_loop(self) -> None: 
        """
        This function loops through the training and testing functions by the number of epochs iterations.
        The train and test function are used back to back per epoch to optimize then perfom a second evalution on the perfomance of the model. 
        """
         
        print(f'Model testing and validation: {self._model.__class__.__name__}')       
            
        for epoch in range(self._epochs):
            print('-- epoch '+str(epoch)) 
            print('Training ...')
            self._train(epoch)
            print('Evaluating ...')
            self._test(epoch)
            
            # print(f"-- learning rate : {self._scheduler.get_last_lr()}")

            
    def _train(self, epoch:int) -> None:
        
        """
        This function trains the model and prints the loss and accuracy of the training sets per epoch.
        
        Args:
            epoch (int): The current epoch number for plotting model performance over number of epochs.
            
        Raises:
            RuntimeError
            AttributeError
            TypeError
            Exception
        """
        
        running_loss_train, accuracy_train, predictions, total_predictions = 0.0, 0.0, 0.0, 0.0

        self._model.train()
        
        try:
            print("Train in train_model")
            for images, camera_length, photon_energy, hit_parameter, _ in self._train_loader: 
                inputs = torch.Tensor(images).to(self._device, dtype=torch.float32)
                cam_len = torch.Tensor(camera_length).to(self._device, dtype=torch.float32).squeeze(1)                    
                phot_en = torch.Tensor(photon_energy).to(self._device, dtype=torch.float32).squeeze(1)                    
                self._optimizer.zero_grad()
                score = self._model(inputs, cam_len, phot_en) 
                truth = hit_parameter.reshape(-1, 1).float().to(self._device)
                
                loss = self._criterion(score, truth)
                loss.backward()
                self._optimizer.step()

                running_loss_train += loss.item()
                
                predictions = (torch.sigmoid(score) > 0.5).long()
                accuracy_train += (predictions == truth).float().sum()
                total_predictions += torch.numel(truth)

            loss_train = running_loss_train / len(self._train_loader)  
            self._plot_train_loss[epoch] = loss_train
            print(f'Train loss: {loss_train}')
            accuracy_train /= total_predictions
            self._plot_train_accuracy[epoch] = accuracy_train
            print(f'Train accuracy: {accuracy_train}')

        except RuntimeError as e:
            print(f"RuntimeError during training: {e}")  
        except AttributeError as e:
            print(f"AttributeError during training: {e}")
        except TypeError as e:
            print(f"TypeError during training: {e}")    
        except Exception as e:
            print(f"An unexpected error occurred during training: {e}")
        
    def _test(self, epoch:int) -> None:
        
        """ 
        This function test the model in evaluation mode and prints the loss and accuracy of the testing sets per epoch.
        
        Args:
            epoch (int): The current epoch number for plotting model performance over number of epochs.
            
        Raises:
            RuntimeError
            AttributeError
            TypeError
            Exception
        """
        
        running_loss_test, accuracy_test, predictions, total = 0.0, 0.0, 0.0, 0.0
 
        self._model.eval()

        try:
            print("Test in train_model")
            with torch.no_grad():
                
                for images, camera_length, photon_energy, hit_parameter, _ in self._test_loader:

                    # inputs = inputs.unsqueeze(1).to(self._device, dtype=torch.float32)
                    inputs = torch.Tensor(images).to(self._device, dtype=torch.float32)
                    cam_len = torch.Tensor(camera_length).to(self._device, dtype=torch.float32).squeeze(1)                    
                    phot_en = torch.Tensor(photon_energy).to(self._device, dtype=torch.float32).squeeze(1)      

                    score = self._model(inputs, cam_len, phot_en)
                    truth = hit_parameter.reshape(-1, 1).float().to(self._device)

                    loss = self._criterion(score, truth)
                    running_loss_test += loss.item()

                    predictions = (torch.sigmoid(score) > 0.5).long()
                    accuracy_test += (predictions == truth).float().sum()
                    total += torch.numel(truth)

            loss_test = running_loss_test / len(self._test_loader)
            self._scheduler.step(loss_test)
            self._plot_test_loss[epoch] = loss_test
            accuracy_test /= total
            self._plot_test_accuracy[epoch] = accuracy_test

            print(f'Test loss: {loss_test}')
            print(f'Test accuracy: {accuracy_test}')

        except RuntimeError as e:
            print(f"RuntimeError during testing: {e}")
        except AttributeError as e:
            print(f"AttributeError during testing: {e}")
        except TypeError as e:
            print(f"TypeError during testing: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during testing: {e}")
        
    def plot_loss_accuracy(self, path:str = None) -> None:
        """ 
        This function plots the loss and accuracy of the training and testing sets per epoch.
        
        Args:
            path (str): The location where the plot should be saved
            
        Raises:
            Exception
        """
        try:
            plt.plot(range(self._epochs), self._plot_train_accuracy, marker='.', color='red')
            plt.plot(range(self._epochs), self._plot_test_accuracy, marker='.', color='orange', linestyle='dashed')
            plt.plot(range(self._epochs), self._plot_train_loss, marker='.', color='blue')
            plt.plot(range(self._epochs), self._plot_test_loss, marker='.', color='teal', linestyle='dashed')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('loss/accuracy')
            plt.title(f'Loss and Accuracy for {self._model.__class__.__name__}')
            plt.legend(['accuracy train', 'accuracy test', 'loss train', 'loss test'])

            if path is not None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'training_loss_accuracy.png'
                plt.savefig(path)
                
                print(f'Loss and accuracy plot saved to: {path}')

        except Exception as e:
            print(f"An error occurred while plotting loss and accuracy: {e}")
        
    def save_model(self, path:str) -> None:
        """
        This function saves the model's state_dict to a specified path. This can be used to load the trained model later.
        Save as .pt file.

        Args:
            path (str): The location wherew the model's state_dict should be saved.
        
        Raises:
            Exception
        """

        try:
            torch.save(self._model.state_dict(), path)
            print(f"Model saved to: {path}")

        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
        
    def get_model(self) -> nn.Module:
        """
        This function returns the trained model obkect. This is to get the trained model to evaluation without having to load the state dict. 

        Returns:
            nn.Module: The trained model object. 
        """
        return self._model
