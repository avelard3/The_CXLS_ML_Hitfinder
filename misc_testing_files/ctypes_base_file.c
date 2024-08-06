#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

// Global variables
hid_t file_id, dataset_id, space_id;
hsize_t dims[3];
int ndims;
unsigned int* data = NULL;

int open_dataset(const char* filename, const char* dataset_name, int* out_dims) {
    // Open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return -1;
    }

    // Open the dataset
    dataset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Could not open dataset: %s\n", dataset_name);
        H5Fclose(file_id);
        return -1;
    }

    // Get the dataspace
    space_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims != 3) {
        fprintf(stderr, "Unexpected number of dimensions: %d\n", ndims);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return -1;
    }
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    // Copy dimensions to output array
    for (int i = 0; i < 3; ++i) {
        out_dims[i] = (int)dims[i];
    }

    return ndims;
}

unsigned int* read_layer(int layer_index, int* layer_size) {
    if (layer_index < 0 || layer_index >= dims[0]) {
        fprintf(stderr, "Invalid layer index: %d\n", layer_index);
        return NULL;
    }

    hsize_t offset[3] = {layer_index, 0, 0};
    hsize_t count[3] = {1, dims[1], dims[2]};
    hid_t mem_space_id = H5Screate_simple(3, count, NULL);
    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Allocate memory for the layer
    *layer_size = dims[1] * dims[2];
    data = (unsigned int*) malloc(*layer_size * sizeof(unsigned int));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        H5Sclose(mem_space_id);
        return NULL;
    }

    // Read the data
    if (H5Dread(dataset_id, H5T_NATIVE_UINT, mem_space_id, space_id, H5P_DEFAULT, data) < 0) {
        fprintf(stderr, "Could not read data\n");
        free(data);
        data = NULL;
    }

    H5Sclose(mem_space_id);
    return data;
}

void free_resources() {
    if (data != NULL) {
        free(data);
        data = NULL;
    }
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}
