import h5py as h5
import numpy as np
import torch
from typing import Optional
from torch.utils.data import Dataset
import datetime
# from .The_CXLS_ML_Hitfinder.src.lib.utils import SpecialCaseFunctions
from . import conf
# from .The_CXLS_ML_Hitfinder.src.lib import read_scattering_matrix
class Paths:
    def __init__(self, list_path: list, h5_location: dict, executing_mode: str, is_multi_event: bool = False) -> None:
        """
        Constructor for Paths class that handles both single and multi-event files.
        Args:
            list_path (list): Path to an lst file with h5 file paths.
            h5_location (dict): The image or hyperparameter path name in the h5 file
            executing_mode (str): if it's running or training mode
            is_multi_event (bool, optional): Flag to distinguish between single and multi-event processing. Defaults to False.
        """
        self._list_path = list_path
        self._h5_location = h5_location
        self._h5_tensor_list, self._h5_attr_list, self._h5_file_list = [], [], []
        self._loaded_h5_tensor = None
        self._h5_file_path = None
        self._open_h5_file = None
        self._attribute_holding = {}
        self._number_of_events = 1
        self._executing_mode = executing_mode
        
        self._image_location = self._h5_location['image']
        self._camera_length_location = self._h5_location['camera length'] 
        self._photon_energy_location = self._h5_location['photon energy']
        self._hit_parameter_location = self._h5_location['peak']
        self.crop_images = False #FIXME
        
        self.possible_image_paths = ['/images/', '/entry/data/data']
        self.possible_camera_length_paths = ['/detector_distance', '/entry/instrument/detector/detector_distance']
        self.possible_photon_energy_paths = ['/photon_energy_eV', '/entry/instrument/beam/incident_wavelength']
        self.possible_hit_parameter_paths = ['/hit/']

        
    def run_paths(self) -> None:
        """
        Calls the function that sets up the files to b read and then maps the dataset to a virtual dataset (VDS)
        """
        self.prepare_file_info()
        self.map_dataset_to_vds()
            
    def prepare_file_info(self) -> None:
        """
        Prepares files to be entered into VDS by checking dimensions/shape and storing this information in a vds file
        """
        # add decorators
        
        try:
            now = datetime.datetime.now()
            formatted_date_time = now.strftime('%m%d%y-%H:%M')
            self.vds_name = f'/scratch/avelard3/all_vds/vds_{formatted_date_time}.h5'
            print(f'Creating vds with name: {self.vds_name}')
                    
            # Dynamically determine the number of images
            file_names_only = []
            #Check dimensions and shape of each h5 file & store in np.array
            with open(self._list_path, 'r') as lst_file: # open lst file
                self._dim_and_shape_list = [] #and use list_name.append([img_shape which has num of img in h5, img_dim]) to add more data
                
                for source_file in lst_file: # open one of h5 files in lst file
                    source_file = source_file.strip()
                    
                    if "master" in source_file.lower():
                        if self._executing_mode == 'training':
                            raise NotImplementedError("Cannot train with master file")
                        continue
                    
                    file_names_only.append(source_file)
                    with h5.File(source_file, 'r') as f: # use h5.File to read the h5 file that you just opened
                        
                        for path in self.possible_image_paths:
                            if path in f:
                                self._image_location = path
                        
                        dataset_shape = f[self._image_location].shape
                        image_file_dim = len(dataset_shape)

                        if image_file_dim == 2:
                            self._dim_and_shape_list.append([1, image_file_dim])
                        elif image_file_dim == 3:
                            self._dim_and_shape_list.append([dataset_shape[0], image_file_dim])
                        else:
                            raise IndexError(f"ERROR: Unexpected image file dimensions, expected shape of 2 or 3, but instead found {image_file_dim}")
                            
                    f.close()
            lst_file.close()
            self._dim_and_shape_array = np.array(self._dim_and_shape_list) #shape is [num_images_in_file, 2 or 3 for multievent]
            self._total_num_images = np.sum(self._dim_and_shape_array[:,0])
            print("The total number of images that will be processed is", self._total_num_images)
            self._height, self._width = conf.required_image_size
            self._image_shape = (self._total_num_images, 1, self._height, self._width)
            self._attr_shape = (self._total_num_images, 1) 
        except Exception as e:
            print(f"An unexpected error occurred while preparing file info for loading paths: {e}")

    def map_dataset_to_vds(self) -> None:
        """
        Maps the images and metadata into a virtual dataset (with different methods depending on training/running and data format)
        """
        try: 
            with h5.File(self.vds_name, 'w') as vds_file: #start creating vds file
                # Virtual layout for images
                            
                self._image_layout = h5.VirtualLayout(shape=self._image_shape, dtype='float32')
                self._camera_length_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
                self._photon_energy_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
                if self._executing_mode == 'training':
                    self._hit_parameter_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
                
                # Loop through each source file and map it to the virtual dataset
                k=0
                master_files_encountered = 0            
                
                with open(self._list_path, 'r') as lst_file: # open list file
                    most_recent_master = None
                    for i, source_file in enumerate(lst_file): # for each file numbered up to i in the list file (aka: for i in range(len(lst_file)): source_file = lst_file[i])
                        self._source_file = source_file.strip() 
                        self.add_file_to_list(self._source_file, i) # add the current file name & number to a list to keep track of images
                        if "master" in self._source_file.lower():
                            most_recent_master = self._source_file
                            master_files_encountered += 1
                            with h5.File(self._source_file, 'r') as f:        
                                for path in self.possible_camera_length_paths:
                                    if path in f:
                                        self._camera_length_location = path                                    
                                for path in self.possible_photon_energy_paths:
                                    if path in f:
                                        self._photon_energy_location = path 
                            continue                    
                        
                        with h5.File(self._source_file, 'r') as f: # using h5.File to read the current h5 file in the list                         
                            # Clarify image location
                            for path in self.possible_image_paths:
                                if path in f:
                                    self._image_location = path                                
                            for path in self.possible_camera_length_paths:
                                if path in f:
                                    self._camera_length_location = path
                            for path in self.possible_photon_energy_paths:
                                if path in f:
                                    self._photon_energy_location = path
                            
                            # Image data source
                            vsource_image = h5.VirtualSource(f[self._image_location])
                            i = i - master_files_encountered
                            if most_recent_master != None:
                                with h5.File(most_recent_master, 'r') as mrm:
                                    vsource_camera_length = h5.VirtualSource(mrm[self._camera_length_location]) 
                                    if "wavelength" in self._photon_energy_location.lower():
                                        scaled_value = 12398.41984 / mrm[self._photon_energy_location][()]
                                        with h5.File("scaled_photon_energy.h5", "w") as f:
                                            f.create_dataset('/photon_energy_eV', data=scaled_value)
                                        vsource_photon_energy = h5.VirtualSource("scaled_photon_energy.h5", '/photon_energy_eV', shape=(1,))
                                    else:
                                        vsource_photon_energy = h5.VirtualSource(mrm[self._photon_energy_location])
                            else:
                                vsource_camera_length = h5.VirtualSource(f[self._camera_length_location]) 
                                if "wavelength" in self._photon_energy_location.lower():
                                    scaled_value = 12398.41984 / f[self._photon_energy_location][()]
                                    with h5.File("scaled_photon_energy.h5", "w") as f:
                                        f.create_dataset('/photon_energy_eV', data=scaled_value)
                                    vsource_photon_energy = h5.VirtualSource("scaled_photon_energy.h5", '/photon_energy_eV', shape=(1,))
                                else:
                                    vsource_photon_energy = h5.VirtualSource(f[self._photon_energy_location])

                            
                            if self._dim_and_shape_array[i,1] == 2: #if it's single event #change to Ks!!!!!!!!!!!!!!!!!!!!!#!
                                print("Single event has not been tested recently")
                                self.add_file_to_list(self._source_file, 1)
                                self._image_layout[i, 0, :, :] = vsource_image
                                self._camera_length_layout[i] = vsource_camera_length
                                
                                self._photon_energy_layout[i] = vsource_photon_energy
                                if self._executing_mode == 'training':
                                    for path in self.possible_hit_parameter_paths:
                                        if path in f:
                                            self._hit_parameter_location = path
                                    
                                    vsource_hit_parameter = h5.VirtualSource(f[self._hit_parameter_location])
                                    self._hit_parameter_layout[i] = vsource_hit_parameter
                                    
                            elif self._dim_and_shape_array[i,1] == 3: # (need to have a metadata check eventually) and self._attr_shape[0] > 1: # if it's multievent with indiv metadata
                                
                                if vsource_image.shape[1] != 512:
                                    #TODO add an option of quadrants 1-4 so that beam stop is not in any pictures later
                                    print("Going to crop to 512x512 from current shape", vsource_image.shape)
                                    #starting point 
                                    shape_of_img = vsource_image.shape
                                    center_x = shape_of_img[1]//2
                                    center_y = shape_of_img[2]//2
                                    vsource_image = vsource_image[:, center_x: (center_x + 512), center_y: (center_y +512)]

                                self._image_layout[k:(k+self._dim_and_shape_array[i,0]), 0, :, :] = vsource_image
                                
                                for j in range(self._dim_and_shape_array[i,0]):
                                    self.add_file_to_list(self._source_file, j+1)
                                if most_recent_master != None:
                                    self._camera_length_layout[i] = vsource_camera_length
                                    self._photon_energy_layout[i] = vsource_photon_energy 
                                else:
                                    self._camera_length_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_camera_length
                                    self._photon_energy_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_photon_energy
                                    
                                if self._executing_mode == 'training':
                                    
                                    for path in self.possible_hit_parameter_paths:
                                        if path in f:
                                            self._hit_parameter_location = path
                                            
                                    vsource_hit_parameter = h5.VirtualSource(f[self._hit_parameter_location])
                                    self._hit_parameter_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_hit_parameter

                                
                            else:
                                print("ERROR: Mapping data to VDS. Likely an issue with metadata")

                            f.close()
                        k += self._dim_and_shape_array[i,0]   #keep track of number of images for mix of single/multievent
                    lst_file.close()
                    
                # Create the VDS for images and metadata in the virtual HDF5 file
                vds_file.create_virtual_dataset('vsource_image', self._image_layout)
                vds_file.create_virtual_dataset('vsource_camera_length', self._camera_length_layout)
                vds_file.create_virtual_dataset('vsource_photon_energy', self._photon_energy_layout)
                if self._executing_mode == 'training':
                    vds_file.create_virtual_dataset('vsource_hit_parameter', self._hit_parameter_layout)
                    
        except Exception as e:
            print(f"An unexpected error occurred while mapping the dataset to VDS: {e}")

    def add_file_to_list(self, numbered_file: str, i: int) -> None:
        """
        Adds h5 file name into list of images for use in output file later
        """
        pic_num = i
        numbered_file =f'{numbered_file}_{str(pic_num)}'
        self._h5_file_list.append(numbered_file)

            
    def get_vds(self) -> str:
        """
        Returns vds
        """
        return self.vds_name
    
    
    def get_file_names(self) -> list: 
        """
        Returns list of file names put into h5 file
        """ 
        return self._h5_file_list
