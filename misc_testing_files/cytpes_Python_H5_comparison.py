import h5py
import hdf5plugin
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import ctypes

def tic():
    global _start_time 
    _start_time = time.perf_counter()

def toc():
    elapsed_time = time.perf_counter() - _start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return elapsed_time

def read_h5_dataset(filename, dataset_name):
    with h5py.File(filename, 'r') as file:
        data = file[dataset_name][:]
        array = np.array(data)
        return array

import ctypes
import numpy as np

def C_read_h5_dataset(filename, dataset_name, lib):
    print('Reading file from C')
    # Define the argument and return types for the read_h5_dataset function
    lib.read_h5_dataset.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.read_h5_dataset.restype = ctypes.POINTER(ctypes.c_uint)  # Update to match unsigned int

    # Define the free_data function
    lib.free_data.argtypes = [ctypes.POINTER(ctypes.c_uint)]  # Update to match unsigned int
    lib.free_data.restype = None
    
    filename_bytes = filename.encode('utf-8')
    dataset_name_bytes = dataset_name.encode('utf-8')
    
    data_size = ctypes.c_int()
    dims_out = (ctypes.c_int * 3)()
    
    print('Calling C function')
    data_ptr = lib.read_h5_dataset(filename_bytes, dataset_name_bytes, ctypes.byref(data_size), dims_out)

    if not data_ptr:
        raise ValueError("Failed to read dataset")

    # Convert the data to a numpy array
    print('Converting data to numpy array')
    array = np.ctypeslib.as_array(data_ptr, shape=(data_size.value,))
    
    # Copy the data to ensure it is owned by Python
    array = np.array(array, copy=True)
    
    # Free the data allocated by the C library
    lib.free_data(data_ptr)
    
    print('Done reading file from C')
    return array


def free_c_array(lib, data_ptr):
    print('freeing C array')
    lib.free_data(data_ptr)

def main():
    print('starting comparison...')
    # filename = '/scratch/sbotha/2024-hitfinder-data/real-data/pk7kev3_11_2768_data_000001.h5'
    filename='/scratch/eseveret/hitfinder_data/dataset_2/images/peaks_water_overlay/01/overlay_img_6keV_clen01_26975.h5'
    dataset = 'entry/data/data'
    
    tic()
    lib = ctypes.CDLL('./lib.so')
    C_np_array = C_read_h5_dataset(filename, dataset, lib)
    print(f'shape of C_np_array: {C_np_array.shape}')
    print(f'First 10 elements of C_np_array: {C_np_array[:10]}')
    toc()

    tic()
    print('reading file from Python')
    with ThreadPoolExecutor() as executor:
        future = executor.submit(read_h5_dataset, filename, dataset)
        python_np_array = future.result()
        print(f'shape of python_np_array: {python_np_array.shape}')
        print(f'First 10 elements of python_np_array: {python_np_array[:10]}')
    toc()

if __name__ == '__main__':
    main()
