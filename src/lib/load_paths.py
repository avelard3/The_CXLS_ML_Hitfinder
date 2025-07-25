import h5py as h5
import numpy as np
import torch
from typing import Optional
from torch.utils.data import Dataset
import datetime
from lib.utils import SpecialCaseFunctions
from . import conf

from scipy.constants import h, c, e
# from .The_CXLS_ML_Hitfinder.src.lib import read_scattering_matrix


class Paths:
    def __init__(self, list_path: list, executing_mode: str, is_multi_event: bool = False) -> None:
        """
        Constructor for Paths class that handles both single and multi-event files.
        Args:
            list_path (list): Path to an lst file with h5 file paths.
            h5_location (dict): The image or hyperparameter path name in the h5 file
            executing_mode (str): if it's running or training mode
            is_multi_event (bool, optional): Flag to distinguish between single and multi-event processing. Defaults to False.
        """
        self._list_path = list_path
        self._h5_tensor_list, self._h5_attr_list, self._h5_file_list = [], [], []
        self._loaded_h5_tensor = None
        self._h5_file_path = None
        self._open_h5_file = None
        self._attribute_holding = {}
        self._number_of_events = 1
        self._executing_mode = executing_mode

        
    def run_paths(self) -> None:
        """
        Calls the function that sets up the files to b read and then maps the dataset to a virtual dataset (VDS)
        """
        self._prepare_file_info()
        self._map_dataset_to_vds()
            
    def _prepare_file_info(self) -> None:
        """
        Prepares files to be entered into VDS by checking dimensions/shape and storing this information in a vds file
        """
        # add decorators
        
        try:
            now = datetime.datetime.now()
            formatted_date_time = now.strftime('%m%d%y-%H:%M')
            self.vds_name = f'{self._executing_mode}_vds_delete_me.h5'
            print(f'Creating vds with name: {self.vds_name}')
                    
            # Dynamically determine the number of images
            file_names_only = []
            #Check dimensions and shape of each h5 file & store in np.array
            with open(self._list_path, 'r') as lst_file: # open lst file
                self._dim_and_shape_list = [] #and use list_name.append([img_shape which has num of img in h5, img_dim]) to add more data
                num_bad_files = 0
                num_complete_files = 0
                num_total_files = 0
                
                for source_file in lst_file: # open one of h5 files in lst file
                    source_file = source_file.strip()
                    num_total_files += 1    
                    if "master" in source_file.lower():
                        num_total_files -=1
                        if self._executing_mode == 'training':
                            raise NotImplementedError("Cannot train with master file")
                        continue
                        
                    try:    
                        with h5.File(source_file, 'r') as f: # use h5.File to read the h5 file that you just opened
                            
                            image_location = self._find_path_in_h5(conf.possible_image_paths, f)
                            
                            dataset_shape = f[image_location].shape
                            image_file_dim = len(dataset_shape)

                            if image_file_dim == 2:
                                self._dim_and_shape_list.append([1, image_file_dim])
                            elif image_file_dim == 3:
                                self._dim_and_shape_list.append([dataset_shape[0], image_file_dim])
                            else:
                                raise IndexError(f"ERROR: Unexpected image file dimensions, expected shape of 2 or 3, but instead found {image_file_dim}")
                                
                        f.close()
                        file_names_only.append(source_file)
                        num_complete_files +=1
                        
                        
                    except (OSError, FileNotFoundError) as e:
                        num_bad_files += 1
                        self._dim_and_shape_list.append([0, 0])
                        print(f"Skipping {source_file} due to an error: {e}")
                        print(f"{num_bad_files} files have been skipped out of the {num_complete_files + num_bad_files} processed so far.")
                    
            lst_file.close()
            print(f"Successfully prepared {num_complete_files} out of the {num_total_files}")
            self._dim_and_shape_array = np.array(self._dim_and_shape_list) #shape is [num_images_in_file, 2 or 3 for multievent]
            self._total_num_images = np.sum(self._dim_and_shape_array[:,0])
            print("The total number of images that will be processed is", self._total_num_images)
            self._height, self._width = conf.required_image_size
            self._image_shape = (self._total_num_images, 1, self._height, self._width)
            self._attr_shape = (self._total_num_images, 1) 
        except Exception as e:
            print(f"An unexpected error occurred while preparing file info for loading paths: {e}")

    def _map_dataset_to_vds(self) -> None:
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
                num_bad_files = 0
                num_complete_files = 0
                num_total_files = 0      
                
                with open(self._list_path, 'r') as lst_file: # open list file
                    most_recent_master = None

                    for i, source_file in enumerate(lst_file): # for each file numbered up to i in the list file (aka: for i in range(len(lst_file)): source_file = lst_file[i])
                        self._source_file = source_file.strip() 
                        num_total_files += 1
                         # add the current file name & number to a list to keep track of images
                        if "master" in self._source_file.lower():
                            most_recent_master = self._source_file
                            num_total_files -=1
                            master_files_encountered += 1
                            with h5.File(self._source_file, 'r') as f: #for master file, figure out where images are stored   
                                camera_length_location = self._find_path_in_h5(conf.possible_camera_length_paths, f) 
                                photon_energy_location = self._find_path_in_h5(conf.possible_photon_energy_paths, f) 
                            f.close()                         
                            continue   
                              
                        try:
                            with h5.File(self._source_file, 'r') as f: # using h5.File to read the current h5 file in the list     
                                
                                if most_recent_master != None:
                                    mrm = h5.File(most_recent_master, 'r')

                                else:
                                    mrm = f
                                                    
                                # Figure out where images are stored
                                image_location = self._find_path_in_h5(conf.possible_image_paths, f) 
                                camera_length_location = self._find_path_in_h5(conf.possible_camera_length_paths, mrm) 
                                photon_energy_location = self._find_path_in_h5(conf.possible_photon_energy_paths, mrm)
                                
                                # Image data source
                                vsource_image = h5.VirtualSource(f[image_location])
                                i = i - master_files_encountered
                                                                
                                vsource_camera_length = h5.VirtualSource(mrm[camera_length_location])
                                vsource_photon_energy = self._define_photon_energy(mrm, photon_energy_location) # Check if it's photon_energy or wavelength and create a Virtual Source out of it

                                ## SINGLE EVENT ##
                                if self._dim_and_shape_array[i,1] == 2: 
                                    print("Single event has not been tested recently")
                                    self.add_file_to_list(self._source_file, 1)
                                    
                                    if vsource_image.shape[1] != 512:
                                        vsource_image = self._crop_image(vsource_image) # Crop the image to the correct size
                                    
                                    self._image_layout[i, 0, :, :] = vsource_image
                                    self._camera_length_layout[i] = vsource_camera_length
                                    self._photon_energy_layout[i] = vsource_photon_energy
                                    
                                    if self._executing_mode == 'training':
                                        hit_parameter_location = self._find_path_in_h5(conf.possible_hit_parameter_paths, f)                                        
                                        vsource_hit_parameter = h5.VirtualSource(f[hit_parameter_location])
                                        self._hit_parameter_layout[i] = vsource_hit_parameter
                                
                                ## MULTI EVENT ##
                                elif self._dim_and_shape_array[i,1] == 3: 

                                    # Add files to list
                                    for j in range(self._dim_and_shape_array[i,0]):
                                        self.add_file_to_list(self._source_file, j+1)
                                    if vsource_image.shape[1] != 512:
                                        vsource_image = self._crop_image(vsource_image) # Crop the image to the correct size
                                    self._image_layout[k:(k+self._dim_and_shape_array[i,0]), 0, :, :] = vsource_image
                                    # Add metadata to VDS (different with and without master file)
                                    if most_recent_master != None:
                                        self._camera_length_layout[i] = vsource_camera_length
                                        self._photon_energy_layout[i] = vsource_photon_energy 
                                    else:
                                        self._camera_length_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_camera_length
                                        self._photon_energy_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_photon_energy
                                    # Add hit parameter to VDS
                                    if self._executing_mode == 'training':
                                        hit_parameter_location = self._find_path_in_h5(conf.possible_hit_parameter_paths, f)        
                                        vsource_hit_parameter = h5.VirtualSource(f[hit_parameter_location])
                                        self._hit_parameter_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_hit_parameter               
                                        
                                else:
                                    print("ERROR: Mapping data to VDS. Likely an issue with metadata")

                                # self.add_file_to_list(self._source_file, i) #FIXME I THINK THIS CAN BE DELETED BUT IDK WHY ITS HERE
                                f.close()
                                mrm.close()
                                num_complete_files +=1
                            k += self._dim_and_shape_array[i,0]   #keep track of number of images for mix of single/multievent
                        
                        except (OSError, FileNotFoundError) as e:
                            num_bad_files += 1
                            print(f"Skipping {source_file} due to an error: {e}")
                            print(f"{num_bad_files} files have been skipped out of the {num_complete_files + num_bad_files} processed so far.")
                    
                    lst_file.close()
                    print(f"Successfully read {num_complete_files} out of the {num_total_files}")
                    
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

    def _find_path_in_h5(self, possible_paths, h5_file):
        for path in possible_paths:
            if path in h5_file:
                return path
        raise KeyError(f"None of these paths found in file: {possible_paths}")
            
    def _define_photon_energy(self, current_file, photon_energy_location):
        if "wavelength" in photon_energy_location.lower():
            scaled_value = SpecialCaseFunctions.incident_photon_wavelength_to_energy(current_file[photon_energy_location][()])
            with h5.File("scaled_photon_energy.h5", "w") as f:
                f.create_dataset('/photon_energy_eV', data=scaled_value)
        return h5.VirtualSource(current_file[photon_energy_location])
    
    def _crop_image(self, vsource_image):
        print(f"Going to crop to 512x512 from current shape {vsource_image.shape} on file {self._source_file}")
        shape_of_img = vsource_image.shape
        center_x = shape_of_img[1]//2
        center_y = shape_of_img[2]//2
        vsource_image = vsource_image[:, center_x: (center_x + 512), center_y: (center_y + 512)]
        return vsource_image