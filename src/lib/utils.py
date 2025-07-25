import torch
import numpy as np
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
    
    
class ModelEvaluationMode:
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
