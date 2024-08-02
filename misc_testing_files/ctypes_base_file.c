#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

void process_chunk(const unsigned int* data_chunk, size_t chunk_size);

unsigned int* read_h5_dataset(const char* filename, const char* dataset_name, int* data_size, int* dims_out) {
    // File identifier
    hid_t file_id, dataset_id, space_id, mem_space_id;
    herr_t status;
    hsize_t dims[3] = {1, 1, 1};  // Initialize to handle up to 3 dimensions
    int ndims;
    unsigned int* data = NULL;

    // Open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        fflush(stdout);
        return NULL;
    }

    // Open the dataset
    dataset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Could not open dataset: %s\n", dataset_name);
        fflush(stdout);
        H5Fclose(file_id);
        return NULL;
    }

    // Get the dataspace
    space_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims < 2 || ndims > 3) {
        fprintf(stderr, "Unexpected number of dimensions: %d\n", ndims);
        fflush(stdout);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    // Print the dimensions
    printf("Dimensions: ");
    for (int i = 0; i < ndims; i++) {
        printf("%llu", (unsigned long long)dims[i]);
        if (i < ndims - 1) {
            printf(" x ");
        }
    }
    printf("\n");
    fflush(stdout);

    // Sanity check for dimensions
    for (int i = 0; i < ndims; i++) {
        if (dims[i] <= 0) {
            fprintf(stderr, "Invalid dimension at index %d: %llu\n", i, (unsigned long long)dims[i]);
            fflush(stdout);
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return NULL;
        }
    }

    // Calculate data size and check for overflow
    unsigned long long total_elements = dims[0] * dims[1];
    if (ndims == 3) {
        total_elements *= dims[2];
    }
    if (total_elements > SIZE_MAX / sizeof(unsigned int)) {
        fprintf(stderr, "Data size overflow\n");
        fflush(stdout);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    *data_size = (int) total_elements;
    printf("Total elements: %d\n", *data_size);
    fflush(stdout);

    // Initialize chunk reading
    hsize_t chunk_dims[3] = {dims[0], dims[1], 1};  // Read one slice at a time for 3D, whole dataset for 2D
    if (ndims == 2) {
        chunk_dims[2] = 1;  // Set third dimension to 1 for 2D datasets
    }

    data = (unsigned int*) malloc(chunk_dims[0] * chunk_dims[1] * chunk_dims[2] * sizeof(unsigned int));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fflush(stdout);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Copy the dimensions out
    for (int i = 0; i < ndims; i++) {
        dims_out[i] = (int)dims[i];
    }
    if (ndims == 2) {
        dims_out[2] = 1;  // Set the third dimension to 1 for consistency
    }
    printf("Data size: %d elements\n", *data_size);
    fflush(stdout);

    // Read data in chunks
    hsize_t offset[3] = {0, 0, 0};
    hsize_t count[3] = {chunk_dims[0], chunk_dims[1], 1};  // Read one slice at a time for 3D, whole dataset for 2D

    mem_space_id = H5Screate_simple(ndims, chunk_dims, NULL);

    for (hsize_t i = 0; i < dims[2]; ++i) {
        offset[2] = i;
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
        status = H5Dread(dataset_id, H5T_NATIVE_UINT32, mem_space_id, space_id, H5P_DEFAULT, data);
        if (status < 0) {
            fprintf(stderr, "Could not read data chunk at index %llu\n", (unsigned long long)i);
            fflush(stdout);
            free(data);
            H5Sclose(mem_space_id);
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return NULL;
        }

        // Process the data chunk
        process_chunk(data, chunk_dims[0] * chunk_dims[1] * chunk_dims[2]);
    }

    // Close/release resources
    printf("Closing/releasing resources\n");
    fflush(stdout);
    H5Sclose(mem_space_id);
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    printf("Resources released\n");
    fflush(stdout);

    return data;
}

// Dummy function to process chunks, replace with actual processing
void process_chunk(const unsigned int* data_chunk, size_t chunk_size) {
    // Example processing function for data chunks
    for (size_t i = 0; i < chunk_size; ++i) {
        // Perform processing on data_chunk[i]
    }
}

void free_data(unsigned int* data) {
    printf("Freeing memory\n");
    fflush(stdout);
    free(data);
}
