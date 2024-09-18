import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from . import conf
import sys


class Data(Dataset):
    
    def __init__(self, classification_data: list, attribute_data: list, h5_file_path: list, use_transform: bool) -> None:
        """
        Initialize the Data object with classification and attribute data.

        Args:
            classification_data (list): List of classification data, that being list of pytorch tensors.
            attribute_data (list): List of attribute data, that being list of metadata dictionaries.
            h5_file_path (list): List of h5 file paths.
            use_transform (bool): Whether you want the transforms to be applied to the images to create more data or not
        """
        self.train_loader = None
        self.test_loader = None
        self.inference_loader = None
        self.image_data = classification_data
        self.meta_data = attribute_data
        self.file_paths = h5_file_path
        self.data = list(zip(self.image_data, self.meta_data, self.file_paths))
        
        self.use_transform = use_transform
        self.transforms = None #initialize
        
        # If transforms will be used, then it creates the pytorch object that will be used to transform future data
        if self.use_transform:
            self.make_transform()
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.image_data)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image data and the metadata at the given index.
        """
                
        # Check if a transform needs to be applied and apply it
        print(f'**********************{self.meta_data}**********************')
        try:
            if self.use_transform:
                image = self.transforms(self.image_data[idx])
                return image, self.meta_data[idx], self.file_paths[idx]
            else:
                return self.data[idx]
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