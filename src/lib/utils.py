import torch
import numpy as np
from scipy.constants import h, c, e

from . import train_model
from . import run_model
from . import conf

class CommonFunctions:
    
    def __init__(self) -> None:
        pass 
    
    
    def load_model_state_dict(self) -> None:
        """
        This function loads in the state dict of a model if provided.
        """
        if self.model_path != 'None':
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
            print(f'There is no model state dict to load into: {self.model.__class__.__name__}')
            

class SpecialCaseFunctions:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def reshape_input_data(data_array: np.ndarray):
        """
        This class reshapes the input data array to the correct dimensions for the model.
        
        Args:
            data_array (np.ndarray): The input data array to be reshaped.

        """
        new_height, new_width = conf.eiger_4m_image_size
        
        batch_size, height, width  = data_array.shape

        if new_height < height or new_width < width:
            # Calculate the center of the images
            center_y, center_x = height // 2, width // 2
            
            # Calculate the start and end indices for the crop
            start_y = center_y - new_height // 2
            end_y = start_y + new_height
            start_x = center_x - new_width // 2
            end_x = start_x + new_width
            
            data_array = data_array[:, start_y:end_y, start_x:end_x]
            
            print(f'Cropped input data array from {height}, {width} to {new_height}, {new_width}.')
            
        if new_height > height or new_width > width:
            current_height, current_width = data_array.shape
            
            # Calculate padding needed for each dimension
            pad_height = (new_height - current_height) // 2
            pad_width = (new_width - current_width) // 2

            # Handle odd differences in desired vs. current size
            pad_height_extra = (new_height - current_height) % 2
            pad_width_extra = (new_width - current_width) % 2

            
            data_array = np.pad(data_array, pad_width=((0,0), (pad_height, pad_height + pad_height_extra), (pad_width, pad_width + pad_width_extra)), mode='constant', constant_values=0) 
            
            print(f'Padded input data array from {height}, {width} to {new_height}, {new_width}.') 
            data_array = np.array(data_array)
        
        return data_array

           
    @staticmethod
    def crop_input_data(data_array: np.ndarray, crop_height, crop_width, height, width) -> np.ndarray:

        # Calculate the center of the images
        center_y, center_x = height // 2, width // 2
        
        # Calculate the start and end indices for the crop
        start_y = center_y - crop_height // 2
        end_y = start_y + crop_height
        start_x = center_x - crop_width // 2
        end_x = start_x + crop_width
        
        data_array = data_array[:, start_y:end_y, start_x:end_x]
        
        print(f'Cropped input data array from {height}, {width} to {crop_width}, {crop_height}.')
        
        return data_array
    
    @staticmethod
    def pad_input_data(self, data_array: np.ndarray) -> np.ndarray:
        
        desired_height, desired_width = self._crop_height, self._crop_width        
        batch_size, current_height, current_width  = self._batch_size, self._height, self._width

        # Calculate padding needed for each dimension
        pad_height = (desired_height - current_height) // 2
        pad_width = (desired_width - current_width) // 2

        # Handle odd differences in desired vs. current size
        pad_height_extra = (desired_height - current_height) % 2
        pad_width_extra = (desired_width - current_width) % 2

        
        data_array = np.pad(data_array, pad_width=((0,0), (pad_width, pad_width + pad_width_extra), (pad_height, pad_height + pad_height_extra)), mode='constant', constant_values=0) 
        
        print(f'Padded input data array from {self._height}, {self._width} to {self._crop_width}, {self._crop_height}.') 
        data_array = np.array(data_array)
        return data_array
    
    
    @staticmethod
    def incident_photon_wavelength_to_energy(wavelength: float) -> float:
        """
        This function takes in the wavelength of an incident photon and returns the energy of the photon on eV (electron volts).

        Args:
            wavelength (float): The wavelength of the incident photon in Angstroms.

        Returns:
            float: _description_
        """
        
        energy_J = h * c / wavelength
        energy_eV = energy_J / e
        
        return energy_eV