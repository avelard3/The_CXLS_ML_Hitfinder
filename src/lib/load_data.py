import h5py as h5
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from . import conf
import sys
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



class Data(Dataset):
    
    def __init__(self, vds_path: str, file_list: list, executing_mode: str, use_transform: bool, master_file: Optional[str] = None,) -> None:
        """
        Initialize the Data object with classification and attribute data.

        Args:
            vds_path (str)
            use_transform (bool): Whether you want the transforms to be applied to the images to create more data or not
        """
        self.train_loader = None
        self.test_loader = None
        self.inference_loader = None
        self.vds_path = vds_path
        self.file_list = file_list
        self.executing_mode = executing_mode
        # add function that does this, make NONE in init
        self.file = h5.File(self.vds_path, 'r')
        self.images = self.file['vsource_image']
        
        self.camera_length = self.file['vsource_camera_length']
        self.photon_energy = self.file['vsource_photon_energy']
        if self.executing_mode == "training":
            self.hit_parameter = self.file['vsource_hit_parameter']
        
        self.use_transform = False
        self._master_file = master_file
        
        # If transforms will be used, then it creates the pytorch object that will be used to transform future data
        if self.use_transform:
            self.make_transform()

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return self.images.shape[0]
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset (tuple of image data and metadata) at the given index.
        """
        x = 0        
        # Check if a transform needs to be applied and apply it
        try:
            self._file_index = self.file_list[idx]
            if self.use_transform:
                print("You tried to use a transform when transforms don't work")
                imgg = self.transforms(self.images[idx])
                if idx == 0 and x==0:
                    print("Creating a plot of one of the images that is being used")
                    self.graph_image(imgg)
                    x+=1
                return imgg, self.camera_length[idx], self.photon_energy[idx], self.hit_parameter[idx], self.file_list[idx] #change

            else:

                #if statement with return only one thing in masterfile metadata #! 
                #*
                if self.executing_mode == "running":
                    self.hit_parameter = np.empty(self.camera_length.shape)

                if self._master_file != None:
                    return self.images[idx], self.camera_length[0], self.photon_energy[0], self.hit_parameter[0], self.file_list[idx]
                else:                    
                    imgg = self.images[idx]
                    if idx == 0 and x==0:
                        self.graph_image(imgg)
                        x+=1
                    return self.images[idx], self.camera_length[idx], self.photon_energy[idx], self.hit_parameter[idx], self.file_list[idx] #change

                #*
        except Exception as e:
            print(f"An unexpected error occurred while getting item at index {idx}: {e} and this is with file {self._file_index}")
            
    def make_transform(self) -> None:
        """
        If the transfom flag is true, this function creates the global variable for the transform for image data. 
        This part doesn't interact with the actual data; it just stores pytorch object data for a future transform.
        """
        self.transforms = transforms.Compose([
            transforms.Resize(200) #Resize transform doesn't work for hitfinder, but transforms in general do work
        ])
        
        
    def graph_image(self,smaller_array):        
        smaller_array = smaller_array[0,:,:]
        fig, ax = plt.subplots()
        heatmap = ax.imshow(smaller_array, norm=colors.SymLogNorm(linthresh=100, linscale=1, base=10), cmap='viridis', origin = 'lower')

        cbar = plt.colorbar(heatmap, ax=ax)
        plt.show()
        plt.savefig("/scratch/avelard3/cxls_hitfinder_joblogs/zseedata_trial5.png")

        
class CreateDataLoader():
    def __init__(self, hitfinder_dataset: Data, batch_size: int) -> None:
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
        
    def get_training_data_loaders(self) -> tuple:
        """
        Get the training and testing data loaders.
        """
        return self._train_loader, self._test_loader
    
    def inference_data_loader(self) -> None: 
        """
        Puts the inference data into a dataloader for batch processing.
        """
        print('Making data loader...')
        try:
            num_items = len(self._hitfinder_dataset)
            if num_items == 0:
                raise ValueError("The dataset is empty.")
            
            try:
                self.inference_loader = DataLoader(self._hitfinder_dataset, batch_size=self._batch_size, shuffle=False, pin_memory=True)
                print('Data loader created.')
            except Exception as e:
                print(f"An error occurred while creating data loaders: {e}")
                return

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def get_inference_data_loader(self) -> DataLoader:
        """
        Returns the inference data loader for putting through the trained model. 
        """
        return self.inference_loader