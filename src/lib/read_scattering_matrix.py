
import json
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import h5py as h5
import matplotlib.colors as colors


class ScatteringMatrix():
    def  __init__(self, geom_file_path:str, data_file_path_name:np.array) -> None:
        """
        Has functions to read the geometry files, calculate the q_vector, create a new array and insert the data into new matrix using that information

        Args:
            file_name (str): name of file that stores geometry information
            file_path (str): path to file that stores geometry information
            data_file_path_name (np.array): all images in an array
        """
        self._current_geom_file_path = geom_file_path

        self.read_geom_file()
        self.calculate_q_vec()
        self.create_new_array()
        self.insert_data_into_new_matrix(data_file_path_name)
        self.graph_first_data_piecetogether()
        #self.graph_heatmap_with_vectors()
        

    def read_geom_file(self):
        """
        Read geometry file information
        """
        
        try:
            with open(self._current_geom_file_path, 'r') as file:
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
            print("The original array shape (num_panels, vector_length)", orig_array_shape)
            self._num_panels = orig_array_shape[0]
            self._vector_length = orig_array_shape[1]
        except Exception as e:
            print(f"An unexpected error occurred while reading the geometry file: {e}")
        
        

    def calculate_q_vec(self):
        """
        Calculate q_vector based on file geometry
        """
                
        # Everything that it multiplied should have the shape (num_panels, vector_length, n_fs_int, n_ss_int)
        
        # Creating a correctly sized array of n_fs and n_ss where the index is equal to the value stored at that index
        # (so you can iterate through all the positions in a n_fs by n_ss panel)
        try: 
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
            
        except Exception as e:
            print(f"An unexpected error occurred while calculating the q-vector: {e}")


    def create_new_array(self):
        """
        Create new array that's the correct size for the new image
        """
        # Calculate the dimensions in real space and pixel space to determine the necessary size array to hold all the panels from the detector
        try:
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
            self._final_array_x_len = int(np.ceil((self._max_x - self._min_x)/self._pixel_length_x)) +2
            self._final_array_y_len = int(np.ceil((self._max_y - self._min_y)/self._pixel_length_y)) +2 
            
            print(f"The final shape of x is {self._final_array_x_len} and the final shape of y is {self._final_array_y_len}")
        except Exception as e:
            print(f"An unexpected error occurred while creating a new array for the scattering matrix: {e}")


    def insert_data_into_new_matrix(self, all_data_array: np.array):
        """
        Entering all the parts of the image data array into a cohesive image

        Args:
            all_data_array (np.array): all images in an array
        """
        
        try:
            self._all_data_array = all_data_array
            print("Array shape:", self._all_data_array.shape)
            print("Array ndim:", self._all_data_array.ndim)

            self._num_trials_in_data = self._all_data_array.shape[0]
            print("num_trials_in_data", self._num_trials_in_data)
            # Create the final array for the pixel data using the dimensions from create_new_array
            self._final_array = np.zeros((self._num_trials_in_data ,self._final_array_y_len, self._final_array_x_len)) # shape (num_trials_in_data (ex 82), fs * num_panels, x or ss)
            print("final array shape", self._final_array.shape)
            # FIXME add an if statement to correlate which dimension or axis goes with fs or ss
            #This is conditional on size and shape of data file
            
            # Splitting array in fs and then ss
            self._all_data_array_split_fs = np.array(np.array_split(self._all_data_array, self._all_data_array.shape[2]/self._n_fs_int, axis = 2)) # shape (self._all_data_array.shape[2]/self._n_fs_int , num_trials_in_data (ex 82) , self._all_data_array[1] , self._n_fs_int)
            print("all data array split fs shape", self._all_data_array_split_fs.shape)
            self._all_data_array_split_ss = np.array(np.array_split(self._all_data_array_split_fs, self._all_data_array.shape[1]/self._n_ss_int, axis = 2)) # shape (self._all_data_array.shape[1]/self._n_ss_int , self._all_data_array.shape[2]/self._n_fs_int , num_trials_in_data (ex 82), self._n_ss_int, self._n_fs_int)
            print("all data array split ss shape", self._all_data_array_split_ss.shape)
            # Reorganiznig array to be (num_trials_in_data, n_ss, n_fs, result of division ss, result of division fs)
            print("before transpose")
            self._all_data_array_split = np.transpose(self._all_data_array_split_ss, (2,3,4,0,1))

            # Reshape data to be (num_trials, fs, ss, num_panels) again
            # The final result is each panel has it's own index with the corresponding data for the pixel and trial number
            print("before reshape")
            self._all_data_array_reshape = np.reshape(self._all_data_array_split, (self._num_trials_in_data, self._n_fs_int, self._n_ss_int, self._all_data_array_split.shape[3] * self._all_data_array_split.shape[4])) # shape (num_trials_in_data (ex 82), y or fs per panel, x or ss per panel, num_panels but calc diff way for check)

            # Find the tv_vec for each panel; tv vec is the lowest, leftmost v_vector (and what we previously were assuming the t-vec was)
            # Need to change the fs and ss vectors into pixels so it can be rearranged. 

            
            self._i_ns = (self._t_vec_arr[:,0] + ((self._max_x - self._min_x)/2))/self._pixel_length_x 
            self._j_ns = (self._t_vec_arr[:,1] + ((self._max_y - self._min_y)/2))/self._pixel_length_y
            print("shape of ins", self._i_ns.shape)
            print("shape of jns", self._j_ns.shape)
            # for each panel, for y to y+fs and x to x+ss of the final pixel data array, add all the data from the particular panel 
            #order of if statements corresponds to the order in which the quadrants of the detector are put together
            #numpy.rot90 default is counterclockwise
            print("the number of panels is", self._num_panels)
            for num in range(self._num_panels):
                print(f"current num is {num} of {self._num_panels}")
                if self._fs_vec_arr[num,0] < 0 and self._ss_vec_arr[num,0] > 0: #fs neg and ss is pos
                    print("top right")
                    #90 deg turn counterclockwise??
                    #top right
                    rot_all_data_array = np.flip(self._all_data_array_reshape[:, :, :, num], axis = 2)
                    print("1a")
                    # self._final_array[:, int(np.ceil(self._j_ns[num] - self._n_fs_int)) : int(np.ceil(self._j_ns[num])), int(np.ceil(self._i_ns[num])) : int(np.ceil(self._i_ns[num] + self._n_ss_int))] = rot_all_data_array
                    self._final_array[:, int(np.ceil(self._j_ns[num] - self._n_fs_int)) : int(np.ceil(self._j_ns[num])), int(np.ceil(self._i_ns[num])) : int(np.ceil(self._i_ns[num] + self._n_ss_int))] = rot_all_data_array
                    print("1b")
                elif self._fs_vec_arr[num,0] > 0 and self._ss_vec_arr[num,0] > 0: #if theyre both positive
                    #no rotation
                    #top left
                    print("top left")
                    self._final_array[:, int(np.ceil(self._j_ns[num])) : int(np.ceil(self._j_ns[num] + self._n_fs_int)), int(np.ceil(self._i_ns[num])) : int(np.ceil(self._i_ns[num] + self._n_ss_int))] = self._all_data_array_reshape[:, :, :, num]
                    print("2b")
                elif self._fs_vec_arr[num,0] > 0 and self._ss_vec_arr[num,0] < 0: #fs pos and ss neg
                    #90 deg turn clockwise
                    #bottom left
                    print("bottom left")
                    rot_all_data_array = np.flip(self._all_data_array_reshape[:, :, :, num], axis = 1)
                    print("3a")
                    self._final_array[:, int(np.ceil(self._j_ns[num])) : int(np.ceil(self._j_ns[num] + self._n_fs_int)), int(np.ceil(self._i_ns[num] - self._n_ss_int)) : int(np.ceil(self._i_ns[num]))] = rot_all_data_array
                    print("3b")
                elif self._fs_vec_arr[num,0] < 0 and self._ss_vec_arr[num,0] < 0: #if theyre both negative
                    #180 rotation
                    #bottom right
                    print("bottom_right")
                    rot_all_data_array = np.flip(self._all_data_array_reshape[:, :, :, num], axis = (1,2))
                    print("4a")
                    self._final_array[:, int(np.ceil(self._j_ns[num] - self._n_fs_int)) : int(np.ceil(self._j_ns[num])), int(np.ceil(self._i_ns[num] - self._n_ss_int)) : int(np.ceil(self._i_ns[num]))] = rot_all_data_array
                    print("4b")
                    
                else:
                    print(f"Error: fs is {self._fs_vec_arr[num,0]} and ss is {self._ss_vec_arr[num,0]}")
                print("just before returning array")
            return self._final_array
                # self._final_array[:, int(np.ceil(self._j_ns[num])) : int(np.ceil(self._j_ns[num] + self._n_fs_int)), int(np.ceil(self._i_ns[num])) : int(np.ceil(self._i_ns[num] + self._n_ss_int))] = self._all_data_array_reshape[:, :, :, num]
        except Exception as e:
            print(f"An unexpected error occurred while inserting data into new matrix: {e}")

       
            
    # Checking and graphing outputs

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
        plt.savefig("/scratch/avelard3/test_scattering_yr_later_try1/graph_first_piecetogether_only2.png")
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
            print("red ss")
        
        for i in range(self._num_panels):
            x = ((self._t_vec_arr[i,0] + 100*self._fs_vec_arr[i,0]) + ((self._max_x - self._min_x)/2))/self._pixel_length_x 
            y = ((self._t_vec_arr[i,1] + 100*self._fs_vec_arr[i,1]) + ((self._max_y - self._min_y)/2))/self._pixel_length_y
            plt.scatter(x,y, color = 'blue')
            print("blue fs")

        print("graphed t vec")
        plt.show()
        plt.savefig("/scratch/avelard3/test_scattering_yr_later_try1/z_best_scattering_and_vector.png")

if __name__ == "__main__":
    open_h5_file = h5.File("/data/bioxfel/data/2020/LCLS-2020-Aug-FrommeP172-P182/data/cheetah/hdf5/r0485-october/mfxp17218-r0485_28.cxi", 'r')
        
    all_data_array = np.array(open_h5_file['entry_1/data_1/data']).astype(np.float32)
    #print("all_data_array from read_scattering_matrix was done. I don't think this should run though", all_data_array.size())
    ScatteringMatrix("/scratch/avelard3/big_files/geom_data/epix10k_geometry.json", all_data_array)
