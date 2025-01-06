import h5py as h5
import numpy as np
import torch
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
        
        #FIXME: should be variables of sbatch scripts
        self._image_location = 'entry/data/data'
        self._camera_length_location = '/instrument/Detector-Distance_mm/'
        self._photon_energy_location = '/photon_energy_eV/'
        self._hit_parameter_location = '/control/hit/'
        
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
        
        # Dynamically determine the number of images
        total_images = 0
        file_names_only = []
        #Check dimensions and shape of each h5 file & store in np.array
        with open(self._list_path, 'r') as lst_file: # open lst file
            self._dim_and_shape_list = [] #and use list_name.append([img_shape which has num of img in h5, img_dim]) to add more data
            for source_file in lst_file: # open one of h5 files in lst file
                source_file = source_file.strip()
                file_names_only.append(source_file)
                with h5.File(source_file, 'r') as f: # use h5.File to read the h5 file that you just opened
                    # Assuming the dataset with images is located at 'entry/data/data' #FIXME this should be an input in the sbatch script
                    dataset_shape = f[self._image_location].shape
                    
                    image_file_dim = len(dataset_shape)
                    print('DATASET SHAPE', dataset_shape)
                    print('DATASET DIM', image_file_dim)
                    
                    if image_file_dim == 2:
                        self._dim_and_shape_list.append([1, image_file_dim])
                    elif image_file_dim == 3:
                        self._dim_and_shape_list.append([dataset_shape[0], image_file_dim])
                    else:
                        print("ERROR: dimensions of dataset must be 2 or 3, but instead it was ", image_file_dim)
                    print(self._dim_and_shape_list)
                    print(f'File {source_file} contains {total_images} images.')
                f.close()
        lst_file.close()
        self._dim_and_shape_array = np.array(self._dim_and_shape_list)
        
        self._total_num_images = np.sum(self._dim_and_shape_array[:,0])
        self._height = 2069 #FIXME
        self._width = 2163 #FIXME
        self._image_shape = (self._total_num_images, 1, self._height, self._width) #getting rid of (num_images, 1, height, width)
        self._attr_shape = (self._total_num_images, 1) #change shape to (1,1)
        
        #! self.add_files_to_list(file_names_only, self._dim_and_shape_array)
        #! need to deal with file list

        
    def map_dataset_to_vds(self) -> None:
        with h5.File(self.vds_name, 'w') as vds_file: #start creating vds file
            # Virtual layout for images
            self._image_layout = h5.VirtualLayout(shape=self._image_shape, dtype='float32')
            self._camera_length_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
            self._photon_energy_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
            if self._executing_mode == 'training':
                self._hit_parameter_layout = h5.VirtualLayout(shape=self._attr_shape, dtype='float32')
            
            # Loop through each source file and map it to the virtual dataset
            with open(self._list_path, 'r') as lst_file: # open list file
                
                for i, source_file in enumerate(lst_file): # for each file numbered up to i in the list file (aka: for i in range(len(lst_file)): source_file = lst_file[i])
                    self._source_file = source_file.strip() 
                    self.add_file_to_list(self._source_file, i) # add the current file name & number to a list to keep track of images
                  
                    with h5.File(self._source_file, 'r') as f: # using h5.File to read the current h5 file in the list 
                        # Image data source
                        # I think you can declare these in the beginning together
                        vsource_image = h5.VirtualSource(f[self._image_location])
                        vsource_camera_length = h5.VirtualSource(f[self._camera_length_location]) 
                        vsource_photon_energy = h5.VirtualSource(f[self._photon_energy_location])
                        
                        self._image_layout[i, 0, :, :] = vsource_image # so if there are multiple images in here, then i think i would need to do things like vsource_image[0], vsource_image[1] etc
                        
#* I think I need a different variable to keep track of the number of which image is being looked at 
                        if self._dim_and_shape_array[i,1] == 2: #if it's single event
                            self.add_file_to_list(self._source_file, 1)
                            self._camera_length_layout[i] = vsource_camera_length
                            self._photon_energy_layout[i] = vsource_photon_energy
                            
                        elif self._dim_and_shape_array[i,1] == 2 and len(vsource_camera_length) == 1: # if it's multievent with shared metadata
                            for j in range(self._dim_and_shape_array[i,0]):
                                print("vsource cam length in load_paths", len(vsource_camera_length))
                                print("i+j in load_paths", (i+j))
                                self.add_file_to_list(self._source_file, j+1)
                                self._camera_length_layout[i+j] = vsource_camera_length
                                self._photon_energy_layout[i+j] = vsource_photon_energy
                                
                        elif self._dim_and_shape_array[i,1] == 2 and len(vsource_camera_length) > 1: # if it's multievent with indiv metadata
                            for j in range(self._dim_and_shape_array[i,0]):
                                print("i+j in load_paths", (i+j))
                                self.add_file_to_list(self._source_file, j+1)
                            
                            print("vsource cam length in load_paths", len(vsource_camera_length))
                            self._camera_length_layout[i] = vsource_camera_length #i think this still needs to be different than single event but idk
                            self._photon_energy_layout[i] = vsource_photon_energy
                            
                        else:
                            print("ERROR: adding metadata to virtual datasets")
                        
                        #if we're training
                        if self._executing_mode == 'training':
                            print("Created hit_parameter VDS")
                            vsource_hit_parameter = h5.VirtualSource(f[self._hit_parameter_location])
                            self._hit_parameter_layout[i] = vsource_hit_parameter

                        f.close()
                lst_file.close()

                
            # Create the VDS for images and metadata in the virtual HDF5 file
            vds_file.create_virtual_dataset('vsource_image', self._image_layout)
            vds_file.create_virtual_dataset('vsource_camera_length', self._camera_length_layout)
            vds_file.create_virtual_dataset('vsource_photon_energy', self._photon_energy_layout)
            if self._executing_mode == 'training':
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