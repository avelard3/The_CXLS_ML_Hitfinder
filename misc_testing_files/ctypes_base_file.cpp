#include <iostream>
#include <vector>
#include <hdf5.h>

extern "C" {
    double* read_h5_dataset(const char* filename, const char* dataset_name, int* data_size, int* dims_out) {
        // File identifier
        hid_t file_id, dataset_id, space_id;
        herr_t status;
        hsize_t dims[3];
        int ndims;
        double* data = nullptr;

        // Open the file
        file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            std::cerr << "Could not open file: " << filename << std::endl;
            return nullptr;
        }

        // Open the dataset
        dataset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
        if (dataset_id < 0) {
            std::cerr << "Could not open dataset: " << dataset_name << std::endl;
            H5Fclose(file_id);
            return nullptr;
        }

        // Get the dataspace
        space_id = H5Dget_space(dataset_id);
        ndims = H5Sget_simple_extent_ndims(space_id);
        if (ndims != 3) {
            std::cerr << "Unexpected number of dimensions: " << ndims << std::endl;
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return nullptr;
        }
        H5Sget_simple_extent_dims(space_id, dims, nullptr);

        // Allocate memory for the data
        *data_size = dims[0] * dims[1] * dims[2];
        data = new double[*data_size];
        if (data == nullptr) {
            std::cerr << "Memory allocation failed" << std::endl;
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return nullptr;
        }

        // Copy the dimensions out
        dims_out[0] = static_cast<int>(dims[0]);
        dims_out[1] = static_cast<int>(dims[1]);
        dims_out[2] = static_cast<int>(dims[2]);

        // Read the data
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        if (status < 0) {
            std::cerr << "Could not read data" << std::endl;
            delete[] data;
            data = nullptr;
        }

        // Close/release resources
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);

        return data;
    }

    void free_data(double* data) {
        delete[] data;
    }
}
