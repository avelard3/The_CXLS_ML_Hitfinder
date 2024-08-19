
import json
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import h5py as h5
import matplotlib.colors as colors

# from lib import utils
# from . import conf


class ScatteringMatrix():
    def  __init__(self, file_name:str, file_path:str) -> None:
        self._current_geom_file =  file_name
        self._current_geom_file_path = file_path

        self.read_geom_file()
        self.calculate_q_vec()
        self.create_new_array()
        self.insert_data_into_new_matrix()
        self.graph_first_data_piecetogether()
        

    def read_geom_file(self):
        with open(self._current_geom_file_path + self._current_geom_file, 'r') as file:
            data = json.load(file)

        # Vectors from geom file

        self._fs_vec = [entry['fs_vec'] for entry in data] # fast scan vector

        self._ss_vec = [entry['ss_vec'] for entry in data] # slow scan vector

        self._t_vec = [entry['t_vec'] for entry in data] # center of first pixel vector


        # Size of detector panel from geom file

        n_fs_list = [entry['n_fs'] for entry in data] # number of pixels in fast-scan direction
        self._n_fs_int = n_fs_list[0]
        n_ss_list = [entry['n_ss'] for entry in data] # number of pixels in slow-scan direction
        self._n_ss_int = n_ss_list[0]

        file.close()

    def calculate_q_vec(self):
        # Vector definitions all based on reborn documentation for detectors

        self._b_vec = np.array([0.0, 0.0, 1.0]) # Default that reborn uses

        # reborn finds photon wavelength  by dividing hc / photon_energy, where hc is the constant h * constant c from scipy constants
        # default photon_energy = 1.602e-15 according to reborn
        # also, in an example from reborn, wavelength = 1.5e-10
        # FIXME: currently setting lambda = 1.5e-10, since reborn used it as a fair number to be used in an examples
        self._wavelength = 1.5e-10 
        self._t_vec_arr = np.array(self._t_vec)
        self._ss_vec_arr = np.array(self._ss_vec)
        self._fs_vec_arr = np.array(self._fs_vec)

        orig_array_shape = self._fs_vec_arr.shape

        self._num_panels = orig_array_shape[0]
        self._vector_length = orig_array_shape[1]
        # FIXME: add a break point if they're different sizes
        n_fs_indices = np.arange(self._n_fs_int).reshape(1,1,-1,1) # Shape (1, 1, n_fs, 1)
        n_ss_indices = np.arange(self._n_ss_int).reshape(1,1,1,-1) # Shape(1, 1, 1, n_ss)

        n_fs_indices_expanded = np.tile(n_fs_indices, (self._num_panels, self._vector_length, 1, self._n_ss_int)) # Shape (num_panels, vector_length, n_fs, n_ss)
        n_ss_indices_expanded = np.tile(n_ss_indices, (self._num_panels, self._vector_length, self._n_fs_int, 1)) # Shape (num_panels, vector_length, n_fs, n_ss)


        # Expand dimensions of fs, ss, and t to align with n_fs_indices and n_ss_indices
        # Expanding to shape (num_panels, size_of_vector = 3, n_fs, n_ss)
        # And t_vec, ss_vec, fs_vec have the size and number of vectors (as well as the value of the vectors)
        # The arrays need to be expanded to test in each panel, and each pixel of that panel, with 3 values defining the vector
        b_vec_expanded = self._b_vec.reshape(1,-1,1,1)
        b_vec_expanded = np.tile(b_vec_expanded, (self._num_panels, 1, self._n_fs_int, self._n_ss_int)) # Shape (num_panels, vector_length, n_fs, n_ss)
        
        fs_expanded = self._fs_vec_arr[:, :, np.newaxis, np.newaxis]
        fs_expanded = np.tile(fs_expanded, (1, 1, self._n_fs_int, self._n_ss_int))  # Shape (num_panels, vector_length, n_fs, n_ss)

        ss_expanded = self._ss_vec_arr[:, :, np.newaxis, np.newaxis]  
        ss_expanded = np.tile(ss_expanded, (1, 1, self._n_fs_int, self._n_ss_int))  # Shape (num_panels, vector_length, n_fs, n_ss)

        t_expanded = self._t_vec_arr[:, :, np.newaxis, np.newaxis]  
        t_expanded = np.tile(t_expanded, (1, 1, self._n_fs_int, self._n_ss_int)) # Shape (num_panels, vector_length, n_fs, n_ss)

        # Perform the calculation for each set of vectors
        # Because of adding the dimensions and ones to everything, this operation works.
        self._v_vec = fs_expanded * n_fs_indices_expanded + ss_expanded * n_ss_indices_expanded + t_expanded  # Shape (num_panels, size_of_vector = 3, n_fs, n_ss)
        self._v_vec = np.array(self._v_vec)
        self._q_vec = ((2*np.pi)/self._wavelength) * ((self._v_vec/npla.norm(self._v_vec)) - b_vec_expanded) 

        print("shape of qvec", self._q_vec.shape)


    def create_new_array(self):
        # For graphing heatmap of lengths of q_vec to check q_vec

        ss_ecul_norm = npla.norm(self._ss_vec, axis=1)
        fs_eucl_norm = npla.norm(self._fs_vec, axis=1)


        # Finding size of array in x and y direction
        # self._pixel_length_x = np.mean(ss_ecul_norm)
        # self._pixel_length_y = np.mean(fs_eucl_norm)
        # print("pixel x",self._pixel_length_x)
        # print("pixel y", self._pixel_length_y)
        
        self._pixel_length_x = 1e-4
        self._pixel_length_y = 1e-4

        self._max_x = np.max(self._v_vec[:,0,:,:]) 
        self._max_y = np.max(self._v_vec[:,1,:,:])
        self._min_x = np.min(self._v_vec[:,0,:,:]) 
        self._min_y = np.min(self._v_vec[:,1,:,:]) 

        self._final_array_x_len = int(np.ceil((self._max_x - self._min_x)/self._pixel_length_x))
        self._final_array_y_len = int(np.ceil((self._max_y - self._min_y)/self._pixel_length_y)) +1 
        
        print("self._final_array_x_len", ((self._final_array_x_len)))
        print("self._final_array_y_len", ((self._final_array_y_len)))


    def insert_data_into_new_matrix(self):
        # FIXME: Temporary reading cxi files because I dont want to deal with other classes
        self._open_h5_file = h5.File('/scratch/sbotha/2024-hitfinder-data/epix10k2M-data/mfxly0020-r0130_294.cxi', 'r')
        
        self._all_data_array = np.array(self._open_h5_file['entry_1/data_1/data']).astype(np.float32)
        self._num_trials_in_data = self._all_data_array.shape[0]
        
        self._final_array = np.zeros((self._num_trials_in_data ,self._final_array_y_len, self._final_array_x_len)) # shape (num_trials_in_data (ex 82), fs * num_panels, x or ss)
        
        print("_all_data_array.shape", self._all_data_array.shape)
        
        # Maybe add an if statement to correlate which dimension or axis goes with fs or ss
        #This is conditional on size and shape of data file
        
        # Splitting array in fs and then ss
        
        #* og, see x1
        # Changing (2,3,4,0,1) into (2,3,4,1,0), makes things worse (see x2)
        # Changing order of splitting fs and ss changes NOTHING (see x3)
        self._all_data_array_split_fs = np.array(np.array_split(self._all_data_array, self._all_data_array.shape[2]/self._n_fs_int, axis = 2)) # shape (self._all_data_array.shape[2]/self._n_fs_int , num_trials_in_data (ex 82) , self._all_data_array[1] , self._n_fs_int)
        print("_all_data_array_split_fs.shape", self._all_data_array_split_fs.shape) 

        self._all_data_array_split_ss = np.array(np.array_split(self._all_data_array_split_fs, self._all_data_array.shape[1]/self._n_ss_int, axis = 2)) # shape (self._all_data_array.shape[1]/self._n_ss_int , self._all_data_array.shape[2]/self._n_fs_int , num_trials_in_data (ex 82), self._n_ss_int, self._n_fs_int)
        print("_all_data_array_split_ss.shape", self._all_data_array_split_ss.shape) 

        
        self._all_data_array_split = np.transpose(self._all_data_array_split_ss, (2,3,4,0,1))


        self._all_data_array_reshape = np.reshape(self._all_data_array_split, (self._num_trials_in_data, self._n_fs_int, self._n_ss_int, self._all_data_array_split.shape[3] * self._all_data_array_split.shape[4])) # shape (num_trials_in_data (ex 82), y or fs per panel, x or ss per panel, num_panels but calc diff way for check)

        
        
        print("self._all_data_array_reshape.shape",self._all_data_array_reshape.shape)
        
        
        # Find the tv_vec for each panel
        #tv vec is the lowest, leftmost v_vector (and what we previously were assuming the t-vec was)
        
        xv_vec = self._v_vec[:,0,:,:]
        yv_vec = self._v_vec[:,1,:,:]
        print("xv_vec.shape", xv_vec.shape)
        print("yv_vec.shape", yv_vec.shape)
        
        
        xv_min = np.min(xv_vec, axis = 1)
        yv_min = np.min(yv_vec, axis = 2)
        print("xv_min.shape", xv_min.shape)
        print("yv_min.shape", yv_min.shape)
        
        # We first get the global minimums across b and c for each a
        min_xv_over_yv = np.min(xv_min, axis=1)  # Shape (a,)
        min_yv_over_xv = np.min(yv_min, axis=1)  # Shape (a,)

        # Combine them into a single array of shape (a, 2)
        tv_vec = np.column_stack((min_xv_over_yv, min_yv_over_xv))
        print("tv_vec. shape", tv_vec.shape)
        print("tv vector", tv_vec)



        i_ns = (tv_vec[:,0] + ((self._max_x - self._min_x)/2))/self._pixel_length_x 
        j_ns = (tv_vec[:,1] + ((self._max_y - self._min_y)/2))/self._pixel_length_y
        print("ins output", i_ns)
        print("jns output", j_ns)        
        print("size of vvec", self._v_vec.shape)
        print("size of tvec array", self._t_vec_arr.shape)
        print("i_ns shape", i_ns.shape)
        print("j_ns shape", j_ns.shape)
        print("tv_vec shape", tv_vec.shape)


        plt.scatter(i_ns,j_ns)
        plt.show()
        plt.savefig("/scratch/avelard3/cxls_hitfinder_joblogs/tvec_dot_real_space_try4.png")
        
        for num in range(self._num_panels):
            print(f"({int(np.ceil(i_ns[num]))}, {int(np.ceil(j_ns[num]))}) - ({int(np.ceil(i_ns[num])) + self._n_ss_int}, {int(np.ceil(j_ns[num])) + self._n_fs_int})")
            
            self._final_array[:, int(np.ceil(j_ns[num])) : int(np.ceil(j_ns[num] + self._n_fs_int)), int(np.ceil(i_ns[num])) : int(np.ceil(i_ns[num] + self._n_ss_int))] = self._all_data_array_reshape[:, :, :, num]
            
            
            
        # self._final_size_array = SpecialCaseFunctions.pad_input_data(self._final_array)
        # smaller_array = self._final_size_array[1,:,:]
        # fig, ax = plt.subplots()
        # heatmap = ax.imshow(smaller_array, norm=colors.SymLogNorm(linthresh=100, linscale=1, base=10), cmap='viridis')

        # cbar = plt.colorbar(heatmap, ax=ax)
        # plt.show()
        # plt.savefig("/scratch/avelard3/cxls_hitfinder_joblogs/y1test_data_piecetogether.png")
        
        
    # checking and graphing outputs

    def graph_relative_t_vec(self, path_and_name_to_save):
        # Creating a graph of where the t vector points relative to other vectors
        for i in range(self._num_panels):
            test= self._t_vec_arr[i,:]
            x,y=test[:2]
            plt.scatter(x,y)
            
        plt.show()
        plt.savefig(__path__)


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
        heatmap = ax.imshow(smaller_array, norm=colors.SymLogNorm(linthresh=100, linscale=1, base=10), cmap='viridis')

        cbar = plt.colorbar(heatmap, ax=ax)
        plt.show()
        plt.savefig("/scratch/avelard3/cxls_hitfinder_joblogs/x5test_data_piecetogether.png")
        
        #z3 and x1 are significant
        
        



if __name__ == "__main__":
    ScatteringMatrix("epix10k_geometry.json", "/scratch/avelard3/The_CXLS_ML_Hitfinder/src/geom_data/")






















# All the different versions of testing if the matrix thing works


# ##################################################################################################################################################################################################################################################################
# # The OG
# # Format the x and y list of values to create the right size aray
# # np.arange creates an array thats the length of n_fs or n_ss with the value at each index equal to that index
# # np.reshape the ones allow numpy to broadcast on the two other inputs that aren't defined by n_fs or n_ss
# n_fs_indices = np.arange(n_fs_int).reshape(-1,1,1) # Shape (n_fs, 1, 1)
# n_ss_indices = np.arange(n_ss_int).reshape(1,-1,1) # Shape(1, n_ss, 1)


# # Expand dimensions of fs, ss, and t to align with n_fs_indices and n_ss_indices
# # Expanding to shape (num_panels, n_fs, n_ss, size_of_vector = 3)
# # And t_vec, ss_vec, fs_vec have the size and number of vectors (as well as the value of the vectors)
# # The arrays need to be expanded to test in each panel, and each pixel of that panel, with 3 values defining the vector
# fs_expanded = fs_vec_arr[:, np.newaxis, np.newaxis, :]  # Shape (num_panels, 1, 1, size_of_vector = 3)
# fs_expanded = np.tile(fs_expanded, (1, n_fs_int, n_ss_int, 1))  # Shape (num_panels, n_fs, n_ss, size_of_vector = 3)

# ss_expanded = ss_vec_arr[:, np.newaxis, np.newaxis, :]  # Shape (num_panels, 1, 1, size_of_vector = 3)
# ss_expanded = np.tile(ss_expanded, (1, n_fs_int, n_ss_int, 1))  # Shape (num_panels, n_fs, n_ss, size_of_vector = 3)

# t_expanded = t_vec_arr[:, np.newaxis, np.newaxis, :]  # Shape (num_panels, 1, 1, size_of_vector = 3)
# t_expanded = np.tile(t_expanded, (1, n_fs_int, n_ss_int, 1))  # Shape (num_panels, n_fs, n_ss, size_of_vector = 3)

# # Perform the calculation for each set of vectors
# # Because of adding the dimensions and ones to everything, this operation works.
# v_vec = fs_expanded * n_fs_indices + ss_expanded * n_ss_indices + t_expanded  # Shape (num_panels, n_fs, n_ss, size_of_vector = 3)

# q_vec = ((2*np.pi)/wavelength) * ((v_vec/npla.norm(v_vec)) - b_vec) # FIXME: lambda and b vector are completely made up

# #?################################################################################################################################################################################################################################################################
# # JUST SWITCHING THE ORDER OF EXPANDED BC ERIC IS PICKY
# # but I think this is going to be wrong because the order of n_fs_indices and n_ss_indices is wrong
# n_fs_indices_v0 = np.arange(n_fs_int).reshape(1,-1,1) # Shape (n_fs, 1, 1)
# n_ss_indices_v0 = np.arange(n_ss_int).reshape(1,1,-1) # Shape(1, n_ss, 1)

# #! I don't understand why b_vec needs to be redefined all of the sudden.... I mean I do, but I don't
# b_vec_expanded_v0 = b_vec.reshape(1,-1,1,1)
# b_vec_expanded_v0 = np.tile(b_vec_expanded_v0, (num_panels, 1, n_fs_int, n_ss_int))


# # Expand dimensions of fs, ss, and t to align with n_fs_indices and n_ss_indices
# # Expanding to shape (num_panels, n_fs, n_ss, size_of_vector = 3)
# # And t_vec, ss_vec, fs_vec have the size and number of vectors (as well as the value of the vectors)
# # The arrays need to be expanded to test in each panel, and each pixel of that panel, with 3 values defining the vector
# fs_expanded_v0 = fs_vec_arr[:, :, np.newaxis, np.newaxis]  # Shape (num_panels, 1, 1, size_of_vector = 3)
# fs_expanded_v0 = np.tile(fs_expanded_v0, (1, 1, n_fs_int, n_ss_int))  # Shape (num_panels, n_fs, n_ss, size_of_vector = 3)

# ss_expanded_v0 = ss_vec_arr[:, :, np.newaxis, np.newaxis]  # Shape (num_panels, 1, 1, size_of_vector = 3)
# ss_expanded_v0 = np.tile(ss_expanded_v0, (1, 1, n_fs_int, n_ss_int))  # Shape (num_panels, n_fs, n_ss, size_of_vector = 3)

# t_expanded_v0 = t_vec_arr[:, :, np.newaxis, np.newaxis]  # Shape (num_panels, 1, 1, size_of_vector = 3)
# t_expanded_v0 = np.tile(t_expanded_v0, (1, 1, n_fs_int, n_ss_int))  # Shape (num_panels, n_fs, n_ss, size_of_vector = 3)

# # Perform the calculation for each set of vectors
# # Because of adding the dimensions and ones to everything, this operation works.
# v_vec_v0 = fs_expanded_v0 * n_fs_indices_v0 + ss_expanded_v0 * n_ss_indices_v0 + t_expanded_v0  # Shape (num_panels, n_fs, n_ss, size_of_vector = 3)

# q_vec_v0 = ((2*np.pi)/wavelength) * ((v_vec_v0/npla.norm(v_vec_v0)) - b_vec_expanded_v0) # FIXME: lambda and b vector are completely made up




# #*################################################################################################################################################################################################################################################################
# # EXPAND N_FS_INDICES AND N_SS_INDICES, and I made everything in the vector shape eric wanted




# #!################################################################################################################################################################################################################################################################
# # Doing this with a for loop, even though it's "bad" 
# #? Making vector_length = 1 and hope that append will insert it in the correct spot
# #this needs to be in the format of (num_panel, vector, n_fs, n_ss)
# result = np.empty((num_panels, vector_length, n_fs_int, n_ss_int))
# for p in range(num_panels):
#     for fs in range(n_fs_int):
#         for ss in range(n_ss_int):
#             equation = fs_vec_arr[p]*fs + ss_vec_arr[p]*ss + t_vec_arr[p]
#             equation = np.array(equation)
#             result[p, :, fs, ss] = equation


# print(result.shape)            
# v_vec_for_loop = result

# v_diff_long = v_vec_v2 - v_vec_for_loop

# print("FORLOOP difference (want this to be zero) (before q, this is v):", np.count_nonzero(v_diff_long))


