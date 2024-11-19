import h5py as h5
import hdf5plugin
import numpy as np
import torch
from queue import Queue
import concurrent.futures
from typing import Optional
from torch.utils.data import Dataset
import datetime
# from .The_CXLS_ML_Hitfinder.src.lib.utils import SpecialCaseFunctions
# from .The_CXLS_ML_Hitfinder.src.lib import conf
# from .The_CXLS_ML_Hitfinder.src.lib import read_scattering_matrix
class Paths:
    def __init__(self, list_path: list, attributes: dict, executing_mode: str, master_file: Optional[str] = None, is_multi_event: bool = False) -> None:
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
        self._h5_tensor_list, self._h5_attr_list, self._h5_file_list = [], [], []
        self._loaded_h5_tensor = None
        self._h5_file_path = None
        self._open_h5_file = None
        self._attribute_holding = {}
        self._number_of_events = 1
        self._is_multi_event = is_multi_event
        self._executing_mode = executing_mode
        
    def run_paths(self) -> None:
        self.set_up_files()
        self.map_dataset_to_vds()
            
    def set_up_files(self) -> None:
        # add decorators
        now = datetime.datetime.now()
        formatted_date_time = now.strftime('%m%d%y-%H:%M')
        self.vds_name = f'vds_{formatted_date_time}.h5'
        print(f'Creating vds with name: {self.vds_name}')
        
        #? I think it would be really nice if we could just give it a folder with files in it and tell it to deal with that, rather than making a list file with all the names in it
        
        #! how do i find how many images exist?!?
        num_images = 3
        #FIXME
        height = 2069
        width = 2163
        self._image_shape = (num_images, 1, height, width)

        if self._master_file != None:
            self._attr_shape = (1,1)
        else:
            self._attr_shape = (num_images, 1) #change shape to (1,1)


        
        
    def map_dataset_to_vds(self) -> None:
        with h5.File(self.vds_name, 'w') as vds_file:
            # Virtual layout for images
            self._image_layout = h5.VirtualLayout(shape=self._image_shape, dtype='float32')
            self._camera_length_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
            self._photon_energy_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
            if self._executing_mode == 'running':
                self._hit_parameter_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
            
            # Loop through each source file and map it to the virtual dataset
            with open(self._list_path, 'r') as lst_file:
                for i, source_file in enumerate(lst_file):
                    self._source_file = source_file.strip() #create a global array
                    self.add_file_to_list(self._source_file, i)
                  
                    with h5.File(self._source_file, 'r') as f:
                        # Image data source
                        vsource_image = h5.VirtualSource(f['entry/data/data'])
                        self._image_layout[i, 0, :, :] = vsource_image  # Map into (1, height, width)

                        if self._attr_shape[0] != 1 and i != 0: #multievent is false? ..... but then where does teh first one go?
                            vsource_camera_length = h5.VirtualSource(f['/instrument/Detector-Distance_mm/'])
                            self._camera_length_layout[i] = vsource_camera_length

                            vsource_photon_energy = h5.VirtualSource(f['/photon_energy_eV/'])
                            self._photon_energy_layout[i] = vsource_photon_energy

                            vsource_hit_parameter = h5.VirtualSource(f['/control/hit/'])
                            if self._executing_mode == 'running':
                                self._hit_parameter_layout[i] = vsource_hit_parameter

                        f.close()
                lst_file.close()
                
            # Create the VDS for images and metadata in the virtual HDF5 file
            vds_file.create_virtual_dataset('vsource_image', self._image_layout)
            vds_file.create_virtual_dataset('vsource_camera_length', self._camera_length_layout)
            vds_file.create_virtual_dataset('vsource_photon_energy', self._photon_energy_layout)
            if self._executing_mode == 'running':
                vds_file.create_virtual_dataset('vsource_hit_parameter', self._hit_parameter_layout)
            
    def add_file_to_list(self, numbered_file: str, i: int) -> None:
        if self._is_multi_event:
            pic_num = i % self._number_of_events
            numbered_file =f'{numbered_file}_{str(pic_num)}'
        self._h5_file_list.append(numbered_file)
            
    def get_vds(self) -> str:
        return self.vds_name
    def get_file_names(self) -> list:  
        return self._h5_file_list