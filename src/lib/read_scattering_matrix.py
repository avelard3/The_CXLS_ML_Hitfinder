
import json
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import h5py as h5
import matplotlib.colors as colors


class ScatteringMatrix():
    def  __init__(self, file_name:str, file_path:str, data_file_path_name:np.array) -> None:
        self._current_geom_file =  file_name
        self._current_geom_file_path = file_path

        self.read_geom_file()
        self.calculate_q_vec()
        self.create_new_array()
        self.insert_data_into_new_matrix(data_file_path_name)
        self.graph_first_data_piecetogether()
        

    def read_geom_file(self):
        with open(self._current_geom_file_path + self._current_geom_file, 'r') as file:
            data = json.load(file)

        # Vectors from geom file

        self._fs_vec = [entry['fs_vec'] for entry in data] # fast scan vector

        self._ss_vec = [entry['ss_vec'] for entry in data] # slow scan vector

        self._t_vec = [entry['t_vec'] for entry in data] # center of first pixel vector


        # Size of detector panel from geom file

        n_fs_list = [entry['n_fs'] for entry in data] # should all be equivalent #FIXME add breakpoint if not equivalent
        self._n_fs_int = n_fs_list[0]  # number of pixels in fast-scan direction
        
        n_ss_list = [entry['n_ss'] for entry in data] # should all be equivalent
        self._n_ss_int = n_ss_list[0] # number of pixels in slow-scan direction

        file.close()
        
        # Vector definitions all based on reborn documentation for detectors

        # reborn finds photon wavelength  by dividing hc / photon_energy, where hc is the constant h * constant c from scipy constants
        # default photon_energy = 1.602e-15 according to reborn, but in example reborn used wavelength = 1.5e-10 #FIXME
        # Likely unimportant because qvec isn't used in anything currently
        self._b_vec = np.array([0.0, 0.0, 1.0]) # Default that reborn uses
        self._wavelength = 1.5e-10 
        self._t_vec_arr = np.array(self._t_vec)
        self._ss_vec_arr = (np.array(self._ss_vec))
        self._fs_vec_arr = (np.array(self._fs_vec))

        orig_array_shape = self._fs_vec_arr.shape # FIXME: add a break point if t,fs,ss are different sizes
        self._num_panels = orig_array_shape[0]
        self._vector_length = orig_array_shape[1]
        
        

    def calculate_q_vec(self):

        
        # The following is made more complicated due to the fact that we're not using for loops
        
        # Everything that it multiplied should have the shape (num_panels, vector_length, n_fs_int, n_ss_int)
        
        # Creating a correctly sized array of n_fs and n_ss where the index is equal to the value stored at that index
        # (so you can iterate through all the positions in a n_fs by n_ss panel)
        
        n_fs_indices = np.arange(self._n_fs_int).reshape(1,1,-1,1) # Shape (1, 1, n_fs, 1)
        n_ss_indices = np.arange(self._n_ss_int).reshape(1,1,1,-1) # Shape(1, 1, 1, n_ss)
        n_fs_indices_expanded = np.tile(n_fs_indices, (self._num_panels, self._vector_length, 1, self._n_ss_int)) # Shape (num_panels, vector_length, n_fs, n_ss)
        n_ss_indices_expanded = np.tile(n_ss_indices, (self._num_panels, self._vector_length, self._n_fs_int, 1)) # Shape (num_panels, vector_length, n_fs, n_ss)


        # Expand dimensions of fs, ss, and t to align with n_fs_indices and n_ss_indices
        # The arrays need to be expanded to find the vector in each panel, and each pixel of that panel, with 3 values defining the vector
        fs_expanded = self._fs_vec_arr[:, :, np.newaxis, np.newaxis]
        fs_expanded = np.tile(fs_expanded, (1, 1, self._n_fs_int, self._n_ss_int))  # Shape (num_panels, vector_length, n_fs, n_ss)

        ss_expanded = self._ss_vec_arr[:, :, np.newaxis, np.newaxis]  
        ss_expanded = np.tile(ss_expanded, (1, 1, self._n_fs_int, self._n_ss_int))  # Shape (num_panels, vector_length, n_fs, n_ss)

        t_expanded = self._t_vec_arr[:, :, np.newaxis, np.newaxis]  
        t_expanded = np.tile(t_expanded, (1, 1, self._n_fs_int, self._n_ss_int)) # Shape (num_panels, vector_length, n_fs, n_ss)
        
        # The b vector also needs to be the correct size for multiplication:
        b_vec_expanded = self._b_vec.reshape(1,-1,1,1)
        b_vec_expanded = np.tile(b_vec_expanded, (self._num_panels, 1, self._n_fs_int, self._n_ss_int)) # Shape (num_panels, vector_length, n_fs, n_ss)

        # Perform the calculation for each set of vectors to find the v vector point to each individual pixel on each individual panel

        self._v_vec = fs_expanded * n_fs_indices_expanded + ss_expanded * n_ss_indices_expanded + t_expanded  # Shape (num_panels, size_of_vector = 3, n_fs, n_ss)
        self._v_vec = (np.array(self._v_vec))
        
        # q vector is graphing it in real space like the reborn documentation shows, but I don't think qvec is important
        self._q_vec = ((2*np.pi)/self._wavelength) * ((self._v_vec/npla.norm(self._v_vec)) - b_vec_expanded) 

        print("shape of qvec", self._q_vec.shape)


    def create_new_array(self):
        # Calculate the dimensions in real space and pixel space to determine the necessary size array to hold all the panels from the detector

        # Find the average length of a pixel in real space "units"
        ss_ecul_norm = npla.norm(self._ss_vec, axis=1)
        fs_eucl_norm = npla.norm(self._fs_vec, axis=1)
        self._pixel_length_x = np.mean(ss_ecul_norm)
        self._pixel_length_y = np.mean(fs_eucl_norm)

        # Find the max and min of x and y values of the v vectors in real space to find the range of real space that all the panels take up
        self._max_x = np.max(self._v_vec[:,0,:,:]) 
        self._max_y = np.max(self._v_vec[:,1,:,:])
        self._min_x = np.min(self._v_vec[:,0,:,:]) 
        self._min_y = np.min(self._v_vec[:,1,:,:]) 

        # Find the range in the x and y direction of the panels and convert it into pixels 
        self._final_array_x_len = int(np.ceil((self._max_x - self._min_x)/self._pixel_length_x)) +1
        self._final_array_y_len = int(np.ceil((self._max_y - self._min_y)/self._pixel_length_y)) +1 
        
        print("self._final_array_x_len", ((self._final_array_x_len)))
        print("self._final_array_y_len", ((self._final_array_y_len)))


    def insert_data_into_new_matrix(self, all_data_array: np.array):        
        self._all_data_array = all_data_array
        self._num_trials_in_data = self._all_data_array.shape[0]
        
        # Create the final array for the pixel data using the dimensions from create_new_array
        self._final_array = np.zeros((self._num_trials_in_data ,self._final_array_y_len, self._final_array_x_len)) # shape (num_trials_in_data (ex 82), fs * num_panels, x or ss)
        
        print("_all_data_array.shape", self._all_data_array.shape)
        
        # FIXME add an if statement to correlate which dimension or axis goes with fs or ss
        #This is conditional on size and shape of data file
        
        # Splitting array in fs and then ss
        self._all_data_array_split_fs = np.array(np.array_split(self._all_data_array, self._all_data_array.shape[2]/self._n_fs_int, axis = 2)) # shape (self._all_data_array.shape[2]/self._n_fs_int , num_trials_in_data (ex 82) , self._all_data_array[1] , self._n_fs_int)
        self._all_data_array_split_ss = np.array(np.array_split(self._all_data_array_split_fs, self._all_data_array.shape[1]/self._n_ss_int, axis = 2)) # shape (self._all_data_array.shape[1]/self._n_ss_int , self._all_data_array.shape[2]/self._n_fs_int , num_trials_in_data (ex 82), self._n_ss_int, self._n_fs_int)

        # Reorganiznig array to be (num_trials_in_data, n_ss, n_fs, result of division ss, result of division fs)
        self._all_data_array_split = np.transpose(self._all_data_array_split_ss, (2,3,4,0,1))

        # Reshape data to be (num_trials, fs, ss, num_panels) again
        # The final result is each panel has it's own index with the corresponding data for the pixel and trial number
        self._all_data_array_reshape = np.reshape(self._all_data_array_split, (self._num_trials_in_data, self._n_fs_int, self._n_ss_int, self._all_data_array_split.shape[3] * self._all_data_array_split.shape[4])) # shape (num_trials_in_data (ex 82), y or fs per panel, x or ss per panel, num_panels but calc diff way for check)

        # Find the tv_vec for each panel; tv vec is the lowest, leftmost v_vector (and what we previously were assuming the t-vec was)
        
        xv_vec = self._v_vec[:,0,:,:]
        yv_vec = self._v_vec[:,1,:,:]

        xv_min = np.min(xv_vec, axis = 1)
        yv_min = np.min(yv_vec, axis = 2)

        # Get the global minimums across xv and yv for each panel
        min_xv_over_yv = np.min(xv_min, axis=1)  # Shape (panels,)
        min_yv_over_xv = np.min(yv_min, axis=1)  # Shape (panels,)

        # Combine them into a single array of shape (panels, 2)
        self._tv_vec = np.column_stack((min_xv_over_yv, min_yv_over_xv))

        # i_ns and j_ns are the bottom leftmost pixel of each panel in pixel space. where i is the leftmost pixel in a panel (x) and j is the bottom (y)
        self._i_ns = (self._tv_vec[:,0] + ((self._max_x - self._min_x)/2))/self._pixel_length_x 
        self._j_ns = (self._tv_vec[:,1] + ((self._max_y - self._min_y)/2))/self._pixel_length_y

        # for each panel, for y to y+fs and x to x+ss of the final pixel data array, add all the data from the particular panel 
        for num in range(self._num_panels):
            # print(f"({int(np.ceil(i_ns[num]))}, {int(np.ceil(j_ns[num]))}) - ({int(np.ceil(i_ns[num])) + self._n_ss_int}, {int(np.ceil(j_ns[num])) + self._n_fs_int})")
            self._final_array[:, int(np.ceil(self._j_ns[num])) : int(np.ceil(self._j_ns[num] + self._n_fs_int)), int(np.ceil(self._i_ns[num])) : int(np.ceil(self._i_ns[num] + self._n_ss_int))] = self._all_data_array_reshape[:, :, :, num]

    # Checking and graphing outputs
    def graph_padded_data(self):
        
        reshaped = ReshapeData(self._final_array)
        self._final_size_array = reshaped.result_array
        
        smaller_array = self._final_size_array[1,:,:]
        fig, ax = plt.subplots()
        heatmap = ax.imshow(smaller_array, norm=colors.SymLogNorm(linthresh=100, linscale=1, base=10), cmap='viridis')

        cbar = plt.colorbar(heatmap, ax=ax)
        plt.show()
        plt.savefig("/scratch/avelard3/cxls_hitfinder_joblogs/zfinal_data_array_padded.png")
        print("Created padded graph")

    def graph_heatmap_lengths(self, path_and_name_to_save):
        # Display heatmap
        # Length of q vector at each pixel/panel
        q_eucl_norm = npla.norm(self._q_vec, axis=1)
        
        fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
        axes = axes.flatten()

        for i in range(self._num_panels):
            matrix_2d = q_eucl_norm[i, :, :]
            ax = axes[i]
            cax = ax.imshow(matrix_2d, cmap='viridis', interpolation='nearest')

        plt.tight_layout()
        plt.savefig(path_and_name_to_save)

    def graph_first_data_piecetogether(self):
        smaller_array = self._final_array[1,:,:]
        fig, ax = plt.subplots()
        heatmap = ax.imshow(smaller_array, norm=colors.SymLogNorm(linthresh=100, linscale=1, base=10), cmap='viridis', origin = 'lower')

        cbar = plt.colorbar(heatmap, ax=ax)
        plt.show()
        plt.savefig("/scratch/avelard3/cxls_hitfinder_joblogs/zfinal_data_array931240.png")
        print("Created graph of all panels")

    def graph_relative_t_vec(self, path_and_name_to_save):
        # Creating a graph of where the t vector points relative to other vectors
        for i in range(self._num_panels):
            test= self._t_vec_arr[i,:]
            x,y=test[:2]
            plt.scatter(x,y)
            
        plt.show()
        plt.savefig(__path__)
        
    def graph_heatmap_with_vectors(self):
        smaller_array = self._final_array[1,:,:]
        fig, ax = plt.subplots()
        heatmap = ax.imshow(smaller_array, norm=colors.SymLogNorm(linthresh=100, linscale=1, base=10), cmap='viridis', origin = 'lower')

        cbar = plt.colorbar(heatmap, ax=ax)
        
        print("Graphed panels")
        
        for i in range(self._num_panels):
            x = self._i_ns[i]
            y = self._j_ns[i]
            plt.scatter(x,y, color = 'yellow')
            print("yellow")

            
        for i in range(self._num_panels):
            x = (self._t_vec_arr[i,0] + ((self._max_x - self._min_x)/2))/self._pixel_length_x 
            y = (self._t_vec_arr[i,1] + ((self._max_y - self._min_y)/2))/self._pixel_length_y
            plt.scatter(x,y, color = 'green')
            print("green")
        
        
        for i in range(self._num_panels):
            x = ((self._t_vec_arr[i,0] + 100*self._ss_vec_arr[i,0]) + ((self._max_x - self._min_x)/2))/self._pixel_length_x 
            y = ((self._t_vec_arr[i,1] + 100*self._ss_vec_arr[i,1]) + ((self._max_y - self._min_y)/2))/self._pixel_length_y
            plt.scatter(x,y, color = 'red')
            print("red")
        
        for i in range(self._num_panels):
            x = ((self._t_vec_arr[i,0] + 100*self._fs_vec_arr[i,0]) + ((self._max_x - self._min_x)/2))/self._pixel_length_x 
            y = ((self._t_vec_arr[i,1] + 100*self._fs_vec_arr[i,1]) + ((self._max_y - self._min_y)/2))/self._pixel_length_y
            plt.scatter(x,y, color = 'blue')
            print("blue")
        
        # for i in range(self._num_panels):
        #     x = self._i_ns[i] + 50
        #     y = self._j_ns[i]
        #     plt.scatter(x,y, color = 'cyan')
        #     print("cyan")
        
        # for i in range(self._num_panels):
        #     x = self._i_ns[i]
        #     y = self._j_ns[i] + 50
        #     plt.scatter(x,y, color = 'magenta')
        #     print("magenta")

        print("graphed t vec")
        plt.show()
        plt.savefig("/scratch/avelard3/cxls_hitfinder_joblogs/zt00_data_array.png")
   
# Methods for padding annd cropping data (the class is defined below but its called in ScatteringMatrix graph_padded_data())     
class ReshapeData():
    def __init__(self,data_array: np.ndarray):
        """
        This class reshapes the input data array to the correct dimensions for the model.
        
        Args:
            data_array (np.ndarray): The input data array to be reshaped.

        """
        eiger_4m_image_size = (2163, 2069)
        self._crop_height, self._crop_width = eiger_4m_image_size
        
        self._batch_size, self._height, self._width  = data_array.shape
        
        self.data_array = data_array
        
        new_data_array = data_array
        if self._crop_height < self._height or self._crop_width < self._width:
            self.result_array = self.crop_input_data(self.data_array)
        elif self._crop_height > self._height or self._crop_width > self._width:
            self.result_array = self.pad_input_data(self.data_array)
        else:
            self.result_array = self.data_array

           
    
    def crop_input_data(self, data_array: np.ndarray) -> np.ndarray:

        # Calculate the center of the images
        center_y, center_x = self._height // 2, self._width // 2
        
        # Calculate the start and end indices for the crop
        start_y = center_y - self._crop_height // 2
        end_y = start_y + self._crop_height
        start_x = center_x - self._crop_width // 2
        end_x = start_x + self._crop_width

        #? Do we need to worry about odd numbers (when %2 != 0)
        
        data_array = data_array[:, start_y:end_y, start_x:end_x]
        
        print(f'Cropped input data array from {self._height}, {self._width} to {self._crop_width}, {self._crop_height}.')
        
        return data_array
    
    def pad_input_data(self, data_array: np.ndarray) -> np.ndarray:
        
        desired_height, desired_width = self._crop_height, self._crop_width        
        batch_size, current_height, current_width  = self._batch_size, self._height, self._width

        # Calculate padding needed for each dimension
        pad_height = (desired_height - current_height) // 2
        pad_width = (desired_width - current_width) // 2

        # Handle odd differences in desired vs. current size
        pad_height_extra = (desired_height - current_height) % 2
        pad_width_extra = (desired_width - current_width) % 2

        
        data_array = np.pad(data_array, pad_width=((0,0), (pad_width, pad_width + pad_width_extra), (pad_height, pad_height + pad_height_extra)), mode='constant', constant_values=0) 
        
        print(f'Padded input data array from {self._height}, {self._width} to {self._crop_width}, {self._crop_height}.') 
        data_array = np.array(data_array)
        return data_array


if __name__ == "__main__":
    open_h5_file = h5.File('/scratch/sbotha/2024-hitfinder-data/epix10k2M-data/mfxly0020-r0130_294.cxi', 'r')
        
    all_data_array = np.array(open_h5_file['entry_1/data_1/data']).astype(np.float32)
    ScatteringMatrix("epix10k_geometry.json", "/scratch/avelard3/big_files/geom_data/", all_data_array)
