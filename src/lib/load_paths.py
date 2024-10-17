import h5py as h5
import hdf5plugin
import numpy as np
import torch
from queue import Queue
import concurrent.futures
from typing import Optional
from torch.utils.data import Dataset
# from .The_CXLS_ML_Hitfinder.src.lib.utils import SpecialCaseFunctions
# from .The_CXLS_ML_Hitfinder.src.lib import conf
# from .The_CXLS_ML_Hitfinder.src.lib import read_scattering_matrix
class Paths:
    def __init__(self, list_path: list, attributes: dict, master_file: Optional[str] = None, is_multi_event: bool = False) -> None:
        """
        Constructor for Paths class that handles both single and multi-event files.
        Args:
            list_path (list): Path to an lst file with h5 file paths.
            attributes (dict): Dictionary with metadata attributes to read from hdf5 files.
            master_file (Optional[str], optional): Path to master file for metadata. Defaults to None.
            is_multi_event (bool, optional): Flag to distinguish between single and multi-event processing. Defaults to False.
        """
        self._list_path = list_path
        self._attributes = attributes
        self._master_file = master_file
        self._master_dict = {}
        self._h5_files = Queue()
        self._h5_tensor_list, self._h5_attr_list, self._h5_file_list = [], [], []
        self._loaded_h5_tensor = None
        self._h5_file_path = None
        self._open_h5_file = None
        self._attribute_holding = {}
        self._number_of_events = 1
        self._is_multi_event = is_multi_event
        
    def map_dataset_to_vds(self) -> None:
        #! come up with fancy name
        self.vds_name = 'name_this_something.h5'
        #! how do i find how many images exist?!?
        num_images = 3
        #FIXME
        height = 2069
        width = 2163
        image_shape = (num_images, 1, height, width)
        attr_shape = (num_images, 1)
        
        with h5.File(self.vds_name, 'w') as vds_file:
            # Virtual layout for images
            image_layout = h5.VirtualLayout(shape=image_shape, dtype='float32')
            # Virtual layout for metadata #FIXME for all attributes
            camera_length_layout = h5.VirtualLayout(shape=attr_shape, dtype='float32')
            photon_energy_layout = h5.VirtualLayout(shape=attr_shape, dtype='float32')
            hit_parameter_layout = h5.VirtualLayout(shape=attr_shape, dtype='float32')
            
            # Loop through each source file and map it to the virtual dataset
            with open(self._list_path, 'r') as lst_file:
                for i, source_file in enumerate(lst_file):
                    source_file = source_file.strip()
                    
                    
                    self._h5_file_list.append(source_file) # adding file to list of files
                    with h5.File(source_file, 'r') as f:
                        # Image data source
                        vsource_image = h5.VirtualSource(f['entry/data/data'])
                        image_layout[i, 0, :, :] = vsource_image  # Map into (1, height, width)
                        print(f'image_layout load_paths map_dataset_to_vds {image_layout.shape}')
                        #print(f'specific image_layout load_paths map_dataset_to_vds {image_layout[i, 0, :, :]}')
                        #^ it didn't like this line of code... you can't do that with virtual data sets?
                        #camera length or detector distance
                        vsource_camera_length = h5.VirtualSource(f['/instrument/Detector-Distance_mm/'])
                        camera_length_layout[i] = vsource_camera_length
                        print(f'camera_length_layout load_paths map_dataset_to_vds {camera_length_layout}')
                        #print(f'specific camera_length_layout load_paths map_dataset_to_vds {camera_length_layout[i]}')
                        vsource_photon_energy = h5.VirtualSource(f['/photon_energy_eV/'])
                        photon_energy_layout[i] = vsource_photon_energy
                        print(f'photon_energy_layout load_paths map_dataset_to_vds {photon_energy_layout}')
                        #print(f'specific photon_energy_layout load_paths map_dataset_to_vds {photon_energy_layout[i]}')
                        vsource_hit_parameter = h5.VirtualSource(f['/control/hit/'])
                        hit_parameter_layout[i] = vsource_hit_parameter
                        print(f'hit_parameter_layout load_paths map_dataset_to_vds {hit_parameter_layout}')
                        #print(f'specific hit_parameter_layout load_paths map_dataset_to_vds {hit_parameter_layout[i]}')
                lst_file.close()
                
            # Create the VDS for images and metadata in the virtual HDF5 file
            vds_file.create_virtual_dataset('vsource_image', image_layout)
            vds_file.create_virtual_dataset('vsource_camera_length', camera_length_layout)
            vds_file.create_virtual_dataset('vsource_photon_energy', photon_energy_layout)
            vds_file.create_virtual_dataset('vsource_hit_parameter', hit_parameter_layout)
            
    def get_vds(self) -> str:
        return self.vds_name
    def get_file_names(self) -> list:
        return self._h5_file_list