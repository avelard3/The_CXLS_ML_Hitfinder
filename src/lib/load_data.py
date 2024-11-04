import h5py as h5
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from . import conf
import sys
from typing import Optional


class Data(Dataset):
    
    def __init__(self, vds_path: str, file_list: list, use_transform: bool, master_file: Optional[str] = None,) -> None:
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
        # add function that does this, make NONE in init
        self.file = h5.File(self.vds_path, 'r')
        self.images = self.file['vsource_image']
        
        self.camera_length = self.file['vsource_camera_length']
        self.photon_energy = self.file['vsource_photon_energy']
        self.hit_parameter = self.file['vsource_hit_parameter']
        
        
        self.use_transform = use_transform
        self.transforms = None #initialize
        self._master_file = master_file
        
        # If transforms will be used, then it creates the pytorch object that will be used to transform future data
        if self.use_transform:
            self.make_transform()
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.images.shape[0]
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image data and the metadata at the given index.
        """
                
        # Check if a transform needs to be applied and apply it
        try:
            if self.use_transform:
                print("You tried to use a transform when transforms don't work")
                image = self.transforms(self.image_data[idx])
                return image, self.meta_data[idx], self.file_paths[idx]
            else:
                print(f'shape self.images[idx] in load_data getitem {self.images[idx].shape}')
                print(f'shape self.imagesload_data getitem {self.images.shape}')
                print(f'shape self.camera_length[idx] in load_data getitem {self.camera_length[idx].shape}')
                print(f'shape self.camera_length in load_data getitem {self.camera_length.shape}')
                print(f'shape self.hit_parameter[idx] in load_data getitem {self.hit_parameter[idx].shape}')
                print(f'shape self.hit_parameter in load_data getitem {self.hit_parameter.shape}')

                #if statement with return only one thing in masterfile metadata #! 
                #*
                if self._master_file != None:
                    return self.images[idx], self.camera_length[0], self.photon_energy[0], self.hit_parameter[0], self.file_list[idx]
                else:
                    return self.images[idx], self.camera_length[idx], self.photon_energy[idx], self.hit_parameter[idx], self.file_list[idx] #change
                #*
        except Exception as e:
            print(f"An unexpected error occurred while getting item at index {idx}: {e}")
            
    def make_transform(self) -> None:
        """
        If the transfom flag is true, this function creates the global variable for the transform for image data. 
        This part doesn't interact with the actual data; it just stores pytorch object data for a future transform.
        """
        self.transforms = transforms.Compose([
            transforms.Resize(300) #Resize transform doesn't work for hitfinder, but transforms in general do work
        ])

        
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

        Args:
            _batch_size (int): The size of the batches to be used by the data loaders.
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

        Returns:
            tuple: A tuple containing the training and testing data loaders.
        """
        return self._train_loader, self._test_loader
    
    def inference_data_loader(self) -> None: 
        """
        Puts the inference data into a dataloader for batch processing.

        Args:
            _batch_size (int): The size of the batches to be used by the data loaders.
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
        This function returns the inference data loader.

        Returns:
            DataLoader: The data loader for putting through the trained model. 
        """
        return self.inference_loader