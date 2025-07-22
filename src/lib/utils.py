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