#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

double* read_h5_dataset(const char* filename, const char* dataset_name, int* data_size) {
    // File identifier
    hid_t file_id, dataset_id, space_id;
    herr_t status;
    hsize_t dims[3];
    int ndims;
    double* data = NULL;

    // Open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return NULL;
    }

    // Open the dataset
    dataset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Could not open dataset: %s\n", dataset_name);
        H5Fclose(file_id);
        return NULL;
    }

    // Get the dataspace
    space_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims != 3) {
        fprintf(stderr, "Unexpected number of dimensions: %d\n", ndims);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    // Allocate memory for the data
    *data_size = dims[0] * dims[1] * dims[2];
    data = (double*) malloc(*data_size * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Read the data
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    if (status < 0) {
        fprintf(stderr, "Could not read data\n");
        free(data);
        data = NULL;
    }

    // Close/release resources
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return data;
}

void free_data(double* data) {
    free(data);
}
