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
    def __init__(self, list_path: list, executing_mode: str) -> None:
        """Constructor for Paths class

        This function instantiates all variables necessary for Paths class

        Args:
            list_path (list): Path to an lst file with h5 file paths.
            h5_location (dict): The image or hyperparameter path name in the h5 file
            executing_mode (str): Assert whether hitfinder is in training mode or inference (running) mode

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
        """ FIXME Fancy Title with Better Words
        
        This function calls the functions that set up the files to be read and then maps the data to a virtual dataset (VDS)
        """
        self._prepare_file_info()
        self._map_dataset_to_vds()
            
    def _prepare_file_info(self) -> None:
        """Pre-processing file info for input into VDS.
        
        This function checks and stores dimensions/shape of image and metadata and storing this information in a vds file
        
        Raises:
            OSError & FileNotFoundError: If there is an issue reading a particular file
            Exception: Other
        """
        # add decorators
        
        try:
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
        """Create a virtual dataset to store image and metadata
        
        This function maps the images and metadata into a virtual dataset (with different methods depending on training/running and data format)
        Creates a Virtual Layout based on dimensions from _prepare_file_info and creates a Virtual Source based on data, maps Virtual Source to Virtual Layout then creates the virtual dataset.
        
        Raises:
            OSError & FileNotFoundError: If there is an issue reading a particular file
            Exception: Other
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
                                    self._add_file_to_list(self._source_file, 1)
                                    
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
                                        self._add_file_to_list(self._source_file, j+1)
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

    def _add_file_to_list(self, numbered_file: str, pic_num: int) -> None:
        """Add h5 file name to list
        
        This function adds h5 file name into list of images for use in output file later, 
        and it includes number at the end to show which image in file is viewed for multievent files
        
        Args:
            numbered_file (str): The current file being added.
            pic_num (int): The image number that is currently being added
        """
        
        numbered_file =f'{numbered_file}_{str(pic_num)}'
        self._h5_file_list.append(numbered_file)


    def _find_path_in_h5(self, possible_paths: list, h5_file) -> str: #FIXME input type
        """
        This function looks for path in h5 file by iterating through a list of possible paths
        
        Args:
            possible_paths (list: str): List of strings of file paths where data is normally stored in h5 file
            h5_file (FIXME): h5 file that is currently being read
            
        Returns:
            path (str): The path to where the metadata is found in the h5 file
            
        Raises:
            KeyError: If none of the paths in the list are in the h5 file
        """
        
        for path in possible_paths:
            if path in h5_file:
                return path
        raise KeyError(f"None of these paths found in file: {possible_paths}")
            
    def _define_photon_energy(self, current_file, photon_energy_location: str): #FIXME input type and output type
        """
        This function checks if given wavelength to change to photon energy, and creates Virtual Source from photon energy 
        
        Args:
            current_file (FIXME): The h5 file that is currently being read
            photon_energy_location (str): Place in h5 file where photon energy or wavelength is stored
            
        Returns:
            h5py.VirtualSource: A virtual source of the photon energy of the current file
        """
        if "wavelength" in photon_energy_location.lower():
            scaled_value = SpecialCaseFunctions.incident_photon_wavelength_to_energy(current_file[photon_energy_location][()])
            with h5.File("scaled_photon_energy.h5", "w") as f:
                f.create_dataset('/photon_energy_eV', data=scaled_value)
        return h5.VirtualSource(current_file[photon_energy_location])
    
    def _crop_image(self, vsource_image): #FIXME input type and output type
        """
        Crops Virtual Source image to make sure all images are 512x512
        
        Args:
            vsource_image (h5py.VirtualSource): The VirtualSource of the image that needs to be cropped
            
        Returns:
            h5py.VirtualSource: The cropped image
        """
        print(f"Going to crop to 512x512 from current shape {vsource_image.shape} on file {self._source_file}")
        shape_of_img = vsource_image.shape
        center_x = shape_of_img[1]//2
        center_y = shape_of_img[2]//2
        vsource_image = vsource_image[:, center_x: (center_x + 512), center_y: (center_y + 512)]
        return vsource_image
    
    
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