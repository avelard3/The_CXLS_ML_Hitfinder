import ctypes
import numpy as np

# Define the DatasetInfo structure
class DatasetInfo(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint)),
        ("data_size", ctypes.c_int),
        ("dims", ctypes.c_int * 3),
        ("ndims", ctypes.c_int)
    ]

# Load the shared library
lib = ctypes.CDLL('./libh5dataset.so')

# Define the argument and return types for the read_h5_dataset function
lib.read_h5_dataset.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.read_h5_dataset.restype = ctypes.POINTER(DatasetInfo)

# Define the argument and return types for the free_dataset_info function
lib.free_dataset_info.argtypes = [ctypes.POINTER(DatasetInfo)]
lib.free_dataset_info.restype = None

def read_h5_dataset(filename, dataset_name):
    filename_bytes = filename.encode('utf-8')
    dataset_name_bytes = dataset_name.encode('utf-8')

    info_ptr = lib.read_h5_dataset(filename_bytes, dataset_name_bytes)
    
    if not info_ptr:
        raise ValueError("Failed to read dataset")

    info = info_ptr.contents

    # Convert the data to a numpy array
    array = np.ctypeslib.as_array(info.data, shape=(info.data_size,))
    
    # Reshape the array to the correct dimensions
    array = array.reshape(info.dims[0], info.dims[1], info.dims[2])

    # Copy the data to ensure it is owned by Python
    array = np.array(array, copy=True)

    # Free the data allocated by the C++ library
    lib.free_dataset_info(info_ptr)

    return array, (info.dims[0], info.dims[1], info.dims[2])

def main():
    # filename = "/scratch/sbotha/2024-hitfinder-data/real-data/pk7kev3_11_2768_data_000001.h5"
    filename = '/scratch/eseveret/hitfinder_data/dataset_2/images/peaks_water_overlay/01/overlay_img_6keV_clen01_26975.h5'
    dataset_name = "entry/data/data"
    data, dims = read_h5_dataset(filename, dataset_name)
    print(f"Data shape: {dims}")
    print(data)

if __name__ == "__main__":
    main()
