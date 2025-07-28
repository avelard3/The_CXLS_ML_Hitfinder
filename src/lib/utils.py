import torch
import numpy as np
import torch.nn as nn
from . import models as m
import inspect



from scipy.constants import h, c, e

from . import train_model
from . import run_model
from . import conf

class SpecialCaseFunctions:
    
    def __init__(self) -> None:
        pass 
    
    @staticmethod
    def incident_photon_wavelength_to_energy(wavelength: float) -> float:
        """
        This function takes in the wavelength of an incident photon and returns the energy of the photon on eV (electron volts).

        Args:
            wavelength (float): The wavelength of the incident photon in Angstroms.

        Returns:
            photon energy (float)L 
        """
        
        energy_J = h * c / wavelength
        energy_eV = energy_J / e
        
        return energy_eV
    
    
class LoadModel:
    def __init__(self) -> None:
        pass 
    
    @staticmethod
    def do_evaluation(model, device, data_loader):
        with torch.no_grad():
                
            for images, camera_length, photon_energy, hit_parameter, _ in data_loader:

                inputs = torch.Tensor(images).to(device, dtype=torch.float32)
                cam_len = torch.Tensor(camera_length).to(device, dtype=torch.float32).squeeze(1)                    
                phot_en = torch.Tensor(photon_energy).to(self._device, dtype=torch.float32).squeeze(1)      
                score = model(inputs, cam_len, phot_en)
                truth = hit_parameter.reshape(-1, 1).float().to(self._device)
    
    @staticmethod
    def make_model_instance(model_arch, model_in) -> None:
        """
        Create an instance of the model class specified by the model architecture.
        """
        try:
            model = getattr(m, model_arch)(model_inputs=model_in)
            print(f'Model object has been created: {model.__class__.__name__}')
            return model
        except AttributeError:
            print(f"Error: Model '{model_arch}' not found in the module.")
            print(f'Available models: {inspect.getmembers(m, inspect.isclass)}')
        except TypeError:
            print(f"Error: '{model_arch}' found in module is not callable.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")            

                
    @staticmethod
    def load_and_return_model(model_path, model, device) -> nn.Module: 
        """
        Load the state dictionary into the model class and prepare it for evaluation.
        """
        try:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            model.to(device)
            print(f'The model state dict has been loaded into: {model.__class__.__name__}')
            return model 
            
        except FileNotFoundError:
            print(f"Error: The file '{model_path}' was not found.")
        except torch.serialization.pickle.UnpicklingError:
            print(f"Error: The file '{model_path}' is not a valid PyTorch model file.")
        except RuntimeError as e:
            print(f"Error: There was an issue loading the state dictionary into the model: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")