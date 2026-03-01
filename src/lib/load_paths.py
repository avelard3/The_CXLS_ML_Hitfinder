import h5py as h5
import numpy as np
import torch
from typing import Optional
from torch.utils.data import Dataset
import datetime
from lib.utils import SpecialCaseFunctions
from . import conf
from . import read_scattering_matrix
import tempfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os 

from scipy.constants import h, c, e
# from .The_CXLS_ML_Hitfinder.src.lib import read_scattering_matrix
#


class Paths:
    def __init__(self, list_path: list, executing_mode: str, path_to_geom: str) -> None:
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
        self._path_to_geom = path_to_geom
        
        if self._path_to_geom == "None":
            self._path_to_geom = None

        
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
                        # if self._executing_mode == 'training':
                        #     raise NotImplementedError("Cannot train with master file")
                        continue
                        
                    try:    
                        with h5.File(source_file, 'r') as f: # use h5.File to read the h5 file that you just opened
                            
                            image_location = self._find_path_in_h5(conf.possible_image_paths, f)
                            
                            dataset_shape = f[image_location].shape
                            image_file_dim = len(dataset_shape)
                            if image_file_dim == 2: #Single Event
                                self._dim_and_shape_list.append([1, image_file_dim])
                            elif image_file_dim == 3: #Multi Event
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
                                i = i - master_files_encountered  
                                image_location = self._find_path_in_h5(conf.possible_image_paths, f) 
                                image_data_holding = f[image_location]
                                if most_recent_master != None:
                                    mrm = h5.File(most_recent_master, 'r')
                                    goal_size = (self._dim_and_shape_array[i,0], 1) ## I think i might not be updating correctly because of the master files #############
                                    current_master_camera_length = mrm[camera_length_location][()]
                                    master_camera_length = np.full(goal_size, current_master_camera_length, dtype='float32')
                                    current_master_photon_energy = mrm[photon_energy_location][()]
                                    master_photon_energy = np.full(goal_size, current_master_photon_energy, dtype='float32')
                                    
                                    with h5.File(f'camera_length{i}.h5', 'w') as f:
                                        dset = f.create_dataset(f'{camera_length_location}', shape=goal_size, dtype='float32')
                                        dset[:] = master_camera_length
                                        vsource_camera_length = h5.VirtualSource(dset)
                                    with h5.File(f'photon_energy{i}.h5', 'w') as f:
                                        dset = f.create_dataset(photon_energy_location, shape=goal_size, dtype='float32')                                        
                                        dset[:] = master_photon_energy
                                        vsource_photon_energy = self._define_photon_energy(dset, photon_energy_location) # Check if it's photon_energy or wavelength and create a Virtual Source out of it
                                
                                else:
                                    camera_length_location = self._find_path_in_h5(conf.possible_camera_length_paths, f) 
                                    vsource_camera_length = h5.VirtualSource(f[camera_length_location])
                                    photon_energy_location = self._find_path_in_h5(conf.possible_photon_energy_paths, f)
                                    vsource_photon_energy = self._define_photon_energy(f, photon_energy_location)
                                                    
                                if self._path_to_geom != None: #If multipanel detector needs to be pieced together
                                    image_data_holding = self._multipanel_to_single(self._path_to_geom, image_data_holding, image_location)
                                vsource_image = h5.VirtualSource(image_data_holding)
                                # i = i - master_files_encountered
                                ## SINGLE EVENT ##
                                if self._dim_and_shape_array[i,1] == 2: 
                                    print("Single event has not been tested recently")
                                    self._add_file_to_list(self._source_file, 1)
                                    
                                    if vsource_image.shape[1] != min(conf.required_image_size):
                                        vsource_image = self._crop_image(vsource_image) # Crop the image to the correct size
                                        
                                    self._image_layout[i, 0, :, :] = vsource_image
                                    self._camera_length_layout[i] = vsource_camera_length
                                    self._photon_energy_layout[i] = vsource_photon_energy
                                    
                                    if self._executing_mode == 'training':
                                        hit_file = f
                                        if most_recent_master != None:
                                            og_filename = self._source_file
                                            og_filename = og_filename.strip()
                                            og_filename = os.path.basename(og_filename)
                                            hit_file = f"/scratch/avelard3/NSLS-2019-August/h5_hits/{og_filename}"#FIXME needs to be input
                                        hit_parameter_location = self._find_path_in_h5(conf.possible_hit_parameter_paths, hit_file) 
                                        with h5.File(self._source_file, 'r') as hf:                                       
                                            vsource_hit_parameter = h5.VirtualSource(hf[hit_parameter_location])
                                        self._hit_parameter_layout[i] = vsource_hit_parameter
                                
                                ## MULTI EVENT ##
                                elif self._dim_and_shape_array[i,1] == 3: 
                                    # Add files to list
                                    for j in range(self._dim_and_shape_array[i,0]):
                                        self._add_file_to_list(self._source_file, j+1)
                                    
                                    #TODO make sure this works. idk why its commented out
                                    # if self._path_to_geom != None:
                                    #     vsource_image = self._multipanel_to_single(self._path_to_geom, vsource_image)
  
                                    if vsource_image.shape[1] != min(conf.required_image_size):
                                        vsource_image = self._crop_image(vsource_image) # Crop the image to the correct size
                                    self._image_layout[k:(k+self._dim_and_shape_array[i,0]), 0, :, :] = vsource_image
                                    # Add metadata to VDS (different with and without master file)
                                    self._camera_length_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_camera_length
                                    self._photon_energy_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_photon_energy
                                    
                                    # Add hit parameter to VDS
                                    if self._executing_mode == 'training':
                                        hit_file = f
                                        if most_recent_master != None:
                                            og_filename = self._source_file
                                            og_filename = og_filename.strip()
                                            og_filename = os.path.basename(og_filename)
                                            hit_file = f"/scratch/avelard3/NSLS-2019-August/h5_hits/{og_filename}"
                                        else:
                                            hit_file = self._source_file
                                        
                                        #TODO: can i combine these two?? with statements

                                        with h5.File(hit_file, 'r') as h5_hit_file:
                                            hit_parameter_location = self._find_path_in_h5(conf.possible_hit_parameter_paths, h5_hit_file)     


                                        with h5.File(hit_file, 'r') as hf:                           
        
                                            vsource_hit_parameter = h5.VirtualSource(hf[hit_parameter_location])

                                        self._hit_parameter_layout[k:(k+self._dim_and_shape_array[i,0]),0] = vsource_hit_parameter               
                                else:
                                    print("ERROR: Mapping data to VDS. Likely an issue with metadata")

                                f.close()
                                if most_recent_master != None:
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


    def _find_path_in_h5(self, possible_paths: list[str], h5_file: h5.File) -> str: 
        """
        This function looks for path in h5 file by iterating through a list of possible paths
        
        Args:
            possible_paths (list[str]): List of strings of file paths where data is normally stored in h5 file
            h5_file (h5.File): h5 file that is currently being read
            
        Returns:
            path (str): The path to where the metadata is found in the h5 file
            
        Raises:
            KeyError: If none of the paths in the list are in the h5 file
        """

        for path in possible_paths:
            if path in h5_file:
                return path
        raise KeyError(f"None of these paths found in file: {possible_paths}")
            
    def _define_photon_energy(self, current_file, photon_energy_location) -> h5.VirtualSource: 
        """
        This function checks if given wavelength to change to photon energy, and creates Virtual Source from photon energy 
        
        Args:
            current_file (h5.File): The h5 file that is currently being read
            photon_energy_location (str): Place in h5 file where photon energy or wavelength is stored
            
        Returns:
            h5py.VirtualSource: A virtual source of the photon energy of the current file
        """
        if "wavelength" in photon_energy_location.lower():
            
            #scaled_value = SpecialCaseFunctions.incident_photon_wavelength_to_energy(current_file)
            dset = current_file[:]
            energy_J = h * c / dset
            energy_eV = energy_J / e
            
            with h5.File("scaled_photon_energy.h5", "w") as f:
                f.create_dataset('photon_energy', data=energy_eV)
                return h5.VirtualSource(f['photon_energy'])
        return h5.VirtualSource(current_file[photon_energy_location])
    
    def _crop_image(self, vsource_image: h5.VirtualSource) -> h5.VirtualSource: 
        """
        Crops Virtual Source image to make sure all images are min(conf.required_image_size)xmin(conf.required_image_size)
        
        Args:
            vsource_image (h5py.VirtualSource): The VirtualSource of the image that needs to be cropped
            
        Returns:
            h5py.VirtualSource: The cropped image
        """
        shape_of_img = vsource_image.shape
        print(f"Going to crop to min(conf.required_image_size)xmin(conf.required_image_size) from current shape {shape_of_img} on file {self._source_file}")
        center_x = shape_of_img[1]//2
        center_y = shape_of_img[2]//2
        vsource_image = vsource_image[:, center_x: (center_x + min(conf.required_image_size)), center_y: (center_y + min(conf.required_image_size))]
        # self.graph_image(vsource_image, 2)
        return vsource_image
    
    def _multipanel_to_single(self, geom_path:str, image_data, image_location):
        scattering_matrix_manager = read_scattering_matrix.ScatteringMatrix(geom_path, image_data)
        scattering_matrix_manager.read_geom_file()
        scattering_matrix_manager.calculate_q_vec()
        scattering_matrix_manager.create_new_array()
        
        image_data = scattering_matrix_manager.insert_data_into_new_matrix(image_data)
        scattering_matrix_manager.graph_first_data_piecetogether()
        
        temp_data_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp_data_file.close()
        
        self._temp_h5_data_file = h5.File(temp_data_file.name, 'w')
        image_dataset = self._temp_h5_data_file.create_dataset(image_location, data=image_data)
        return image_dataset
        
        
    
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
    
    def graph_image(self, array, number):   
        """Plot an image to see what an example of the data looks like to check orientation"""     
        array = array[0,:,:]
        fig, ax = plt.subplots()
        heatmap = ax.imshow(array, norm=colors.SymLogNorm(linthresh=100, linscale=1, base=10), cmap='viridis', origin = 'lower')

        cbar = plt.colorbar(heatmap, ax=ax)
        plt.show()
        plt.savefig(f"/scratch/avelard3/test_scattering_yr_later_try1/graph_during_load_paths{number}.png") #FIXME: Delete or make path a variable