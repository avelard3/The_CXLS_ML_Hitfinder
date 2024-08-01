#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

unsigned int* read_h5_dataset(const char* filename, const char* dataset_name, int* data_size, int* dims_out) {
    // File identifier
    hid_t file_id, dataset_id, space_id, dtype_id;
    herr_t status;
    hsize_t dims[3];
    int ndims;
    int* data = NULL;

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
    if (ndims != 3) {
        fprintf(stderr, "Unexpected number of dimensions: %d\n", ndims);
        fflush(stdout);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    // Print the dimensions
    printf("Dimensions: %llu x %llu x %llu\n", (unsigned long long)dims[0], (unsigned long long)dims[1], (unsigned long long)dims[2]);
    fflush(stdout);

    // Sanity check for dimensions
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        fprintf(stderr, "Invalid dimensions: %llu x %llu x %llu\n", (unsigned long long)dims[0], (unsigned long long)dims[1], (unsigned long long)dims[2]);
        fflush(stdout);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Calculate data size and check for overflow
    unsigned long long total_elements = dims[0] * dims[1] * dims[2];
    if (total_elements > SIZE_MAX / sizeof(unsigned int)) {
        fprintf(stderr, "Data size overflow\n");
        fflush(stdout);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    *data_size = (int) total_elements;
    printf("Allocating memory for %d elements\n", *data_size);
    fflush(stdout);
    
    data = (unsigned int*) malloc(*data_size * sizeof(unsigned int));
    printf("Data variable assigned\n");
    fflush(stdout);
    printf("Data is %p\n", data);
    fflush(stdout);
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fflush(stdout);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    printf("Data is not NULL\n");
    fflush(stdout);
    // Copy the dimensions out
    dims_out[0] = (int)dims[0];
    dims_out[1] = (int)dims[1];
    dims_out[2] = (int)dims[2];

    // Print the data size
    printf("Data size: %d elements\n", *data_size);
    fflush(stdout);

    if (file_id < 0) {
        fprintf(stderr, "Invalid file identifier\n");
        fflush(stdout);
        return NULL;
    }
    if (dataset_id < 0) {
        fprintf(stderr, "Invalid dataset identifier\n");
        fflush(stdout);
        return NULL;
    }
    if (space_id < 0) {
        fprintf(stderr, "Invalid dataspace identifier\n");
        fflush(stdout);
        return NULL;
    }

    // Validate memory allocation
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fflush(stdout);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // // Validate memory space
    // if (mem_space_id < 0) {
    //     fprintf(stderr, "Invalid memory space identifier\n");
    //     fflush(stdout);
    //     free(data);
    //     H5Sclose(space_id);
    //     H5Dclose(dataset_id);
    //     H5Fclose(file_id);
    //     return NULL;
    // }

    printf("All preconditions for H5Dread are good\n");
    fflush(stdout);

// Double-check the HDF5 datatype
// dtype_id = H5Dget_type(dataset_id);
// if (H5Tequal(dtype_id, H5T_NATIVE_DOUBLE) <= 0) {
//     fprintf(stderr, "Dataset type is not double\n");
//     fflush(stdout);
//     H5Tclose(dtype_id);
//     free(data);
//     H5Sclose(space_id);
//     H5Dclose(dataset_id);
//     H5Fclose(file_id);
//     return NULL;
// }
// H5Tclose(dtype_id);

printf("HDF5 dataset type verified as double\n");
fflush(stdout);

// Double-check the dataspace
if (H5Sget_simple_extent_dims(space_id, dims, NULL) < 0) {
    fprintf(stderr, "Failed to get dataspace dimensions\n");
    fflush(stdout);
    free(data);
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return NULL;
}

printf("Dataspace dimensions verified\n");
fflush(stdout);


    // Read the data
    printf("Reading data...\n");
    fflush(stdout);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    printf("Data read\n");
    fflush(stdout);
    if (status < 0) {
        fprintf(stderr, "Could not read data\n");
        fflush(stdout);
        free(data);
        data = NULL;
    } else {
        printf("Data read successfully\n");
        fflush(stdout);
    }

    // Close/release resources
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return data;
}

void free_data(unsigned int* data) {
    printf("Freeing memory\n");
    fflush(stdout);
    free(data);
}
