import ctypes
import numpy as np
import h5py
import concurrent.futures
import time
from functools import cache

# Define the DatasetInfo structure
class DatasetInfo(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint)),
        ("data_size", ctypes.c_int),
        ("dims", ctypes.c_int * 3),
        ("ndims", ctypes.c_int)
    ]

# Load the shared library
lib = ctypes.CDLL('./lib.so')

# Define the argument and return types for the open_dataset function
lib.open_dataset.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
lib.open_dataset.restype = ctypes.c_int

# Define the argument and return types for the read_layer function
lib.read_layer.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.read_layer.restype = ctypes.POINTER(ctypes.c_uint)

# Define the argument and return types for the free_resources function
lib.free_resources.argtypes = []
lib.free_resources.restype = None

@cache
def open_dataset(filename, dataset_name):
    dims_out = (ctypes.c_int * 3)()
    ndims = lib.open_dataset(filename.encode('utf-8'), dataset_name.encode('utf-8'), dims_out)
    if ndims < 0:
        raise ValueError("Failed to open dataset")
    return ndims, [dims_out[i] for i in range(3)]

@cache
def read_layer(layer_index):
    layer_size = ctypes.c_int()
    data_ptr = lib.read_layer(layer_index, ctypes.byref(layer_size))
    
    if not data_ptr:
        raise ValueError("Failed to read layer")
    
    # Convert the data to a numpy array
    array = np.ctypeslib.as_array(data_ptr, shape=(layer_size.value,))
    
    # Copy the data to ensure it is owned by Python
    array = np.array(array, copy=True)
    
    return array

@cache
def free_resources():
    lib.free_resources()

def tic():
    global _start_time 
    _start_time = time.perf_counter()

def toc():
    elapsed_time = time.perf_counter() - _start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return elapsed_time

@cache
def read_dataset_threading(filename, dataset_name, dims):
    all_data = np.zeros((dims[0], dims[1], dims[2]), dtype=np.uint32)

    def read_layer_thread(layer_index):
        layer = read_layer(layer_index)
        if layer.size != dims[1] * dims[2]:
            raise ValueError(f"Layer {layer_index} size mismatch: expected {dims[1] * dims[2]}, got {layer.size}")
        layer = layer.reshape(dims[1], dims[2])
        all_data[layer_index, :, :] = layer
        print(f"Layer {layer_index} shape: {layer.shape}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_layer_thread, i) for i in range(dims[0])]
        concurrent.futures.wait(futures)
    
    return all_data

@cache
def read_h5_dataset(filename, dataset_name):
    with h5py.File(filename, 'r') as file:
        data = file[dataset_name][:]
        array = np.array(data)
        return array

def main():
    # filename = '/scratch/eseveret/hitfinder_data/dataset_2/images/peaks_water_overlay/01/overlay_img_6keV_clen01_26975.h5'
    filename = "/scratch/sbotha/2024-hitfinder-data/real-data/pk7kev3_11_2768_data_000001.h5"
    dataset_name = "entry/data/data"
    
    print('Using Ctypes with no threading...')
    tic()
    # Using ctypes
    ndims, dims = open_dataset(filename, dataset_name)
    print(f"Number of dimensions: {ndims}")
    print(f"Dimensions: {dims}")

    all_data = np.zeros((dims[0], dims[1], dims[2]), dtype=np.uint32)
    for i in range(dims[0]):
        layer = read_layer(i)
        if layer.size != dims[1] * dims[2]:
            raise ValueError(f"Layer {i} size mismatch: expected {dims[1] * dims[2]}, got {layer.size}")
        layer = layer.reshape(dims[1], dims[2])
        all_data[i, :, :] = layer
    print(f'Shape of all_data: {all_data.shape}')
    print(f'First 10 elements of all_data: {all_data.flat[:10]}')
    print("Time taken with ctypes (sequential):")
    toc()

    free_resources()

    # print('Using Ctypes with threading...')
    # tic()
    # all_data_threaded = read_dataset_threading(filename, dataset_name, dims)
    # print(f'Shape of all_data_threaded: {all_data_threaded.shape}')
    # print(f'First 10 elements of all_data_threaded: {all_data_threaded.flat[:10]}')
    # print("Time taken with ctypes (threading):")
    # toc()

    # free_resources()
    
    # Using h5py
    print('Using h5py with no threading...')
    tic()
    python_np_array = read_h5_dataset(filename, dataset_name)
    print("Time taken with h5py (sequential):")
    print(f'Shape of python_np_array: {python_np_array.shape}')
    print(f'First 10 elements of python_np_array: {python_np_array.flat[:10]}')
    toc()

    print('Using h5py with ThreadPoolExecutor...')
    tic()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(read_h5_dataset, filename, dataset_name)
        try:
            python_np_array_threaded = future.result()
            print("Time taken with h5py (threading):")
            print(f'Shape of python_np_array_threaded: {python_np_array_threaded.shape}')
            print(f'First 10 elements of python_np_array_threaded: {python_np_array_threaded.flat[:10]}')
        except Exception as e:
            print(f"Error in Python reading: {e}")
    toc()

if __name__ == "__main__":
    main()
