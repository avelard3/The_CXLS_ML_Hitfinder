import h5py as h5
import torch
from torch.utils.data import DataLoader, Dataset
from . import conf
import sys
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#
class Data(Dataset):
    
    def __init__(self, file_list: list, executing_mode: str) -> None:
        """
        Initialize the Data object with classification and attribute data.

        Args:
            file_list (list): list of paths to every file being run
            executing mode (str): Indicates whether hitfinder is in training or running mode
        """
        self._run_loader = None
        self._file_list = file_list
        self._executing_mode = executing_mode
        self._vds_path = f'{self._executing_mode}_vds_delete_me.h5'
        
        # add function that does this, make NONE in init
        self.file = h5.File(self._vds_path, 'r')
        self._images = self.file['vsource_image']
        self._camera_length = self.file['vsource_camera_length']
        self._photon_energy = self.file['vsource_photon_energy']
        if self._executing_mode == "training":
            self._hit_parameter = self.file['vsource_hit_parameter']

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return self._images.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[h5.Dataset, h5.Dataset, h5.Dataset, h5.Dataset, list[str]]: 
        """
        Get a sample from the dataset (tuple of image data and metadata) at the given index.
        
        Args:
            idx (int): index of item that is being got
            
        Returns:
            tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset, h5py.Dataset, list[str]]: the data stored at that index for images, camera_length, photon_energy, hit_parameter and the file in the list that is currently being accessed
        """
        x = 0 
        try:
            self._file_index = self._file_list[idx]

            #if statement with return only one thing in masterfile metadata #! 
            
            if idx == 0 and x==0:
                imgg = self._images[idx]
                print("Creating a plot of one of the images that is being used")
                self.graph_image(imgg)
                x+=1
                #*
            if self._executing_mode == "running":
                self._hit_parameter = np.empty(self._camera_length.shape)
            print("the index is", idx)
            print("self._file_list[idx]", self._file_list[idx])
            print("self._camera_length[idx]", self._camera_length[idx])
            print("self._photon_energy[idx]", self._photon_energy[idx])
            
            return self._images[idx], self._camera_length[idx], self._photon_energy[idx], self._hit_parameter[idx], self._file_list[idx] #change

                #*
        except Exception as e:
            print(f"An unexpected error occurred while getting item at index {idx}: {e} and this is with file {self._file_index}")

    def graph_image(self, array):   
        """Plot an image to see what an example of the data looks like to check orientation"""     
        array = array[0,:,:]
        fig, ax = plt.subplots()
        heatmap = ax.imshow(array, norm=colors.SymLogNorm(linthresh=100, linscale=1, base=10), cmap='viridis', origin = 'lower')

        cbar = plt.colorbar(heatmap, ax=ax)
        plt.show()
        plt.savefig("/scratch/avelard3/test_scattering_yr_later_try1/graph_during_load_data.png") #FIXME: Delete or make path a variable

        
class CreateDataLoader():
    def __init__(self, hitfinder_dataset: Data, batch_size: int) -> None:
        
        """Takes a Data object as an input and uses DataLoader to split data into training and testing, or into running. Also includes getters
        """
        
        # Global variables that are inputs
        self._hitfinder_dataset = hitfinder_dataset
        self._batch_size = batch_size
        
        # Other global variables
        self._train_loader = None
        self._test_loader = None
        
    def split_training_data(self) -> None:
        """
        Split the data into training and testing datasets and create data loaders for them.
        """
        try:
            num_items = len(self._hitfinder_dataset)  
            if num_items == 0:
                raise ValueError("The dataset is empty.")
            
            num_train = int(0.8 * num_items)
            num_test = num_items - num_train

            try:
                train_dataset, test_dataset = torch.utils.data.random_split(self._hitfinder_dataset, [num_train, num_test])
            except Exception as e:
                print(f"An error occurred while splitting the dataset: {e}")
                return

            try:
                self._train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True)
                self._test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True)
            except Exception as e:
                print(f"An error occurred while creating data loaders: {e}")
                return
            
            print(f"Train size: {len(train_dataset)}")
            print(f"Test size: {len(test_dataset)}")

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    
    def run_data_loader(self) -> None: 
        """
        Puts the run data into a dataloader for batch processing.
        """
        print('Making data loader...')
        try:
            num_items = len(self._hitfinder_dataset)
            if num_items == 0:
                raise ValueError("The dataset is empty.")
            
            try:
                self._run_loader = DataLoader(self._hitfinder_dataset, batch_size=self._batch_size, shuffle=False, pin_memory=True)
                print('Data loader created.')
            except Exception as e:
                print(f"An error occurred while creating data loaders: {e}")
                return

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def get_training_data_loaders(self) -> tuple:
        """
        Get the training and testing data loaders.
        """
        return self._train_loader, self._test_loader
            
    def get_run_data_loader(self) -> DataLoader:
        """
        Returns the run data loader for putting through the trained model. 
        """
        return self._run_loader