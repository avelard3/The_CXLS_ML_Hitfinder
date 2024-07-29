import h5py
import hdf5plugin
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def read_h5_dataset(filename, dataset_name):
    with h5py.File(filename, 'r') as file:
        data = file[dataset_name][:]
        array = np.array(data)
        return array

def main():
    filename = '/scratch/sbotha/2024-hitfinder-data/real-data/pk7kev3_11_2768_data_000001.h5'
    dataset = 'entry/data/data'

    with ThreadPoolExecutor() as executor:
        future = executor.submit(read_h5_dataset, filename, dataset)

        data = future.result()
        print(data)

if __name__ == '__main__':
    main()
