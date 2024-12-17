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

class TrainModel:
    
    def __init__(self, cfg: dict, attributes: dict, transfer_learning_state_dict: str) -> None:
        """
        This constructor breaks up the training configuration infomation dictionary and h5 metadata key dictionary.
        In addition, a logging object is created and global list are created for storing infomation about the training loss and accuracy. 

        Args:
            cfg (dict): Dictionary containing important information for training including: data loaders, batch size, training device, number of epochs, the optimizer, the scheduler, the criterion, the learning rate, and the model class. Everything besides the data loaders and device are arguments in the sbatch script.
            attributes (dict): Dictionary containing the names of the metadata contained in the h5 image files. These names could change depending on whom created the metadata, so the specific names are arguments in the sbatch script. 
        """
        
        self.camera_length = conf.camera_length_key
        self.photon_energy = conf.photon_energy_key
        self.peak = conf.present_peaks_key
        
        self.model_path = transfer_learning_state_dict

        self.train_loader = None
        self.test_loader = None
        self.batch_size = cfg['batch size']
        self.device = cfg['device']
        self.epochs = cfg['epochs']
        self.optimizer = cfg['optimizer']
        self.scheduler = cfg['scheduler']
        self.criterion = cfg['criterion']
        self.learning_rate = cfg['learning rate']
        self.model = cfg['model']
        
        self.plot_train_accuracy = np.zeros(self.epochs)
        self.plot_train_loss = np.zeros(self.epochs)
        self.plot_test_accuracy = np.zeros(self.epochs)
        self.plot_test_loss = np.zeros(self.epochs)
        
    def assign_new_data(self, train: DataLoader, test: DataLoader) -> None:
        """
        This function assigns new data loaders.

        Args:
            train (DataLoader): The training data loader.
            test (DataLoader): The testing data loader.

        """
        self.train_loader = train
        self.test_loader = test
        
        for _, camera_length, photon_energy, _, _ in self.train_loader: 
            cam_len = np.array(camera_length)                    
            print(cam_len)
            print("shape of cam_len in tensors train train_model, assign_new_data", cam_len.shape)
            phot_en = np.array(photon_energy)                    
            print(phot_en)
            print("shape of photon energy in tensors train train_model, assign_new_data", phot_en.shape)
                    
        for _, camera_length, photon_energy, _, _ in self.test_loader:             
            cam_len = np.array(camera_length)                    
            print(cam_len)
            print("shape of cam_len in tensors test train_model, assign_new_data", cam_len.shape)
            phot_en = np.array(photon_energy)                    
            print(phot_en) 
            print("shape of photon energy in tensors test train_model, assign_new_data", phot_en.shape)
        
    def make_training_instances(self) -> None:
        """
        This function takes the strings from the sbatch script and makes them objects.
        These strings are objects that are needed for the training. Objects declared here are :
        - the model
        - the optimizer
        - the learning rate scheduler
        - the loss criterion
        """
        try:
            self.model = getattr(m, self.model)().to(self.device)
            self.optimizer = getattr(optim, self.optimizer)(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = getattr(lrs, self.scheduler)(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.1)
            self.criterion = getattr(nn, self.criterion)()
            
            print('All training objects have been created.')
            
        except AttributeError as e:
            print(f"AttributeError: {e}")
            if 'nn' in str(e):
                print(f"Error: '{self.model_name}', '{self.criterion_name}' not found in torch.nn")
            elif 'optim' in str(e):
                print(f"Error: '{self.optimizer_name}' not found in torch.optim")
            elif 'lr_scheduler' in str(e):
                print(f"Error: '{self.scheduler_name}' not found in torch.optim.lr_scheduler")
        except TypeError as e:
            print(f"TypeError: {e}")
            if 'nn' in str(e):
                print(f"Error: '{self.model_name}' or '{self.criterion_name}' is not callable")
            elif 'optim' in str(e):
                print(f"Error: '{self.optimizer_name}' is not callable")
            elif 'lr_scheduler' in str(e):
                print(f"Error: '{self.scheduler_name}' is not callable")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def load_model_state_dict(self) -> None:
        """
        This function loads in the state dict of a model if provided.
        """
        if self.model_path != None:
            try:
                state_dict = torch.load(self.transfer_learning_path)
                self.model.load_state_dict(state_dict)
                self.model = self.model.eval() 
                self.model.to(self.device)
                
                print(f'The model state dict has been loaded into: {self.model.__class__.__name__}')
                
            except FileNotFoundError:
                print(f"Error: The file '{self.transfer_learning_path}' was not found.")
            except torch.serialization.pickle.UnpicklingError:
                print(f"Error: The file '{self.transfer_learning_path}' is not a valid PyTorch model file.")
            except RuntimeError as e:
                print(f"Error: There was an issue loading the state dictionary into the model: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        else:
            print(f'There is nno model state dict to load into: {self.model.__class__.__name__}')
    
    def epoch_loop(self) -> None: 
        """
        This function loops through the training and testing functions by the number of epochs iterations.
        The train and test function are used back to back per epoch to optimize then perfom a second evalution on the perfomance of the model. 
        """
         
        print(f'Model testing and validation: {self.model.__class__.__name__}')       
        
        for _, camera_length, photon_energy, _, _ in self.train_loader: 
            cam_len = np.array(camera_length)                    
            print(cam_len)
            print("shape of cam_len in tensors train train_model, just before epoch loop", cam_len.shape)
            phot_en = np.array(photon_energy)                    
            print(phot_en)
            print("shape of photon energy in tensors train train_model, just before epoch loop", phot_en.shape)
                    
        for _, camera_length, photon_energy, _, _ in self.test_loader:             
            cam_len = np.array(camera_length)                    
            print(cam_len)
            print("shape of cam_len in tensors test train_model,, just before epoch loop", cam_len.shape)
            phot_en = np.array(photon_energy)                    
            print(phot_en) 
            print("shape of photon energy in tensors test train_model, just before epoch loop", phot_en.shape)
            
        for epoch in range(self.epochs):
            print('-- epoch '+str(epoch)) 
            print('Training ...')
            self.train(epoch)
            print('Evaluating ...')
            self.test(epoch)
            
            # print(f"-- learning rate : {self.scheduler.get_last_lr()}")

            
    def train(self, epoch:int) -> None:
        
        """
        This function trains the model and prints the loss and accuracy of the training sets per epoch.
        """
        
        running_loss_train, accuracy_train, predictions, total_predictions = 0.0, 0.0, 0.0, 0.0

        self.model.train()
        
        try:
            print("0")
            # so theres an issue right below here
            # so problem is in train_loader and maybe test_loader
            print("train_loader !!!!!!!!!!!!!!!!!!!", self.train_loader)
            for images, camera_length, photon_energy, hit_parameter, _ in self.train_loader: 
                print("0.5")
                print("shape of images at beginning of for loop train", images.shape)
                inputs = torch.Tensor(images).to(self.device, dtype=torch.float32)
                print("shape of images as inputs in tensors train", inputs.shape)
                print(inputs)
                cam_len = torch.Tensor(camera_length).to(self.device, dtype=torch.float32).squeeze()                    
                print(cam_len)
                print("shape of cam_len in tensors train", cam_len.shape)
                phot_en = torch.Tensor(photon_energy).to(self.device, dtype=torch.float32).squeeze()                    
                print(phot_en)
                print("shape of photon energy in tensors train", phot_en.shape)


                self.optimizer.zero_grad()
                print("3")
                score = self.model(inputs, cam_len, phot_en) 
                print("4")
                truth = hit_parameter.reshape(-1, 1).float().to(self.device)
                print("5")
                loss = self.criterion(score, truth)
                print("6") #ran to here
                # self.scaler.scale(loss).backward()
                # print("7")
                # self.scaler.step(self.optimizer)
                # print("8")
                # self.scaler.update()
                print("9")
                running_loss_train += loss.item()
                print("10")
                predictions = (torch.sigmoid(score) > 0.5).long()
                print("11")
                accuracy_train += (predictions == truth).float().sum()
                print("12")
                total_predictions += torch.numel(truth)
                print("13")
            print("14")
            loss_train = running_loss_train / len(self.train_loader)  
            print("15")
            self.plot_train_loss[epoch] = loss_train
            print("16")
            print(f'Train loss: {loss_train}')
            print("17")
            accuracy_train /= total_predictions
            print("18")
            self.plot_train_accuracy[epoch] = accuracy_train
            print("19")
            print(f'Train accuracy: {accuracy_train}')
            print("20")

        except RuntimeError as e:
            print(f"RuntimeError during training: {e}")  
        except AttributeError as e:
            print(f"AttributeError during training: {e}")
        except TypeError as e:
            print(f"TypeError during training: {e}")    
        except Exception as e:
            print(f"An unexpected error occurred during training: {e}")
        
    def test(self, epoch:int) -> None:
        
        """ 
        This function test the model in evaluation mode and prints the loss and accuracy of the testing sets per epoch.
        """
        
        running_loss_test, accuracy_test, predictions, total = 0.0, 0.0, 0.0, 0.0
        for _, camera_length, photon_energy, _, _ in self.train_loader: 
            cam_len = np.array(camera_length)                    
            print(cam_len)
            print("shape of cam_len in tensors train train_model, after self.model.eval", cam_len.shape)
            phot_en = np.array(photon_energy)                    
            print(phot_en)
            print("shape of photon energy in tensors train train_model, after self.model.eval", phot_en.shape)
                    
        for _, camera_length, photon_energy, _, _ in self.test_loader:             
            cam_len = np.array(camera_length)                    
            print(cam_len)
            print("shape of cam_len in tensors test train_model,, after self.model.evalp", cam_len.shape)
            phot_en = np.array(photon_energy)                    
            print(phot_en) 
            print("shape of photon energy in tensors test train_model, after self.model.eval", phot_en.shape)
        self.model.eval()
        for _, camera_length, photon_energy, _, _ in self.train_loader: 
            cam_len = np.array(camera_length)                    
            print(cam_len)
            print("shape of cam_len in tensors train train_model, after self.model.eval", cam_len.shape)
            phot_en = np.array(photon_energy)                    
            print(phot_en)
            print("shape of photon energy in tensors train train_model, after self.model.eval", phot_en.shape)
                    
        for _, camera_length, photon_energy, _, _ in self.test_loader:             
            cam_len = np.array(camera_length)                    
            print(cam_len)
            print("shape of cam_len in tensors test train_model,after self.model.eval", cam_len.shape)
            phot_en = np.array(photon_energy)                    
            print(phot_en) 
            print("shape of photon energy in tensors test train_model, after self.model.eval", phot_en.shape)
        try:
            print("test_loader !!!!!!!!!!!!!!!!!!!", self.test_loader)
            with torch.no_grad():
                for images, camera_length, photon_energy, hit_parameter, _ in self.test_loader:
                    # inputs = inputs.unsqueeze(1).to(self.device, dtype=torch.float32)
                    print("shape of images at beginning of for loop test", images.shape)
                    inputs = torch.Tensor(images).to(self.device, dtype=torch.float32)
                    print("shape of images as inputs in tensors test", inputs.shape)
                    print(inputs)
                    cam_len = torch.Tensor(camera_length).to(self.device, dtype=torch.float32).squeeze()                    
                    print(cam_len)
                    print("shape of cam_len in tensors test", cam_len.shape) # so this is empty 12/17 3:33 (& so is photon energy)
                    phot_en = torch.Tensor(photon_energy).to(self.device, dtype=torch.float32).squeeze()                    
                    print(phot_en)
                    print("shape of photon energy in tensors test", phot_en.shape)
                    print("a")
                    score = self.model(inputs, cam_len, phot_en)
                    print("b")
                    truth = hit_parameter.reshape(-1, 1).float().to(self.device)
                    print("c")
                    loss = self.criterion(score, truth)
                    print("d")
                    running_loss_test += loss.item()
                    print("e")
                    predictions = (torch.sigmoid(score) > 0.5).long()
                    print("f")
                    accuracy_test += (predictions == truth).float().sum()
                    print("g")
                    total += torch.numel(truth)
                    print("h")

            loss_test = running_loss_test / len(self.test_loader)
            print("m")
            self.scheduler.step(loss_test)
            print("n")
            self.plot_test_loss[epoch] = loss_test
            print("p")
            accuracy_test /= total
            print("q")
            self.plot_test_accuracy[epoch] = accuracy_test

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
        """
        try:
            plt.plot(range(self.epochs), self.plot_train_accuracy, marker='o', color='red')
            plt.plot(range(self.epochs), self.plot_test_accuracy, marker='o', color='orange', linestyle='dashed')
            plt.plot(range(self.epochs), self.plot_train_loss, marker='o', color='blue')
            plt.plot(range(self.epochs), self.plot_test_loss, marker='o', color='teal', linestyle='dashed')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('loss/accuracy')
            plt.title(f'Loss and Accuracy for {self.model.__class__.__name__}')
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
            path (str): Path to save the model's state_dict.
        """

        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to: {path}")

        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
        
    def get_model(self) -> nn.Module:
        """
        This function returns the trained model obkect. This is to get the trained model to evaluation without having to load the state dict. 

        Returns:
            nn.Module: The trained model object. 
        """
        return self.model
