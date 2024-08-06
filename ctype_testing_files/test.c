#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <hdf5_hl.h>

// Structure to hold the file and dataset names
typedef struct {
    char *filename;
    char *dataset_name;
} thread_data_t;

// Function to read the HDF5 dataset
void *read_h5_dataset(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    hid_t file_id, dataset_id, datatype, dataspace;
    herr_t status;
    size_t type_size;

    // Open the file
    file_id = H5Fopen(data->filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error opening file: %s\n", data->filename);
        return NULL;
    }
    printf("File opened successfully.\n");

    // Open the dataset
    dataset_id = H5Dopen2(file_id, data->dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening dataset: %s\n", data->dataset_name);
        H5Fclose(file_id);
        return NULL;
    }
    printf("Dataset opened successfully.\n");

    // Get the datatype and dataspace
    datatype = H5Dget_type(dataset_id);
    if (datatype < 0) {
        fprintf(stderr, "Error getting datatype.\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    type_size = H5Tget_size(datatype);
    dataspace = H5Dget_space(dataset_id);
    if (dataspace < 0) {
        fprintf(stderr, "Error getting dataspace.\n");
        H5Tclose(datatype);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Get the number of elements
    hsize_t dims[2];
    int ndims = H5Sget_simple_extent_dims(dataspace, dims, NULL);
    if (ndims < 0) {
        fprintf(stderr, "Error getting dataspace dimensions.\n");
        H5Sclose(dataspace);
        H5Tclose(datatype);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    hsize_t num_elements = dims[0] * dims[1];
    printf("Number of elements: %llu\n", (unsigned long long)num_elements);

    // Allocate memory for the data
    void *data_buffer = malloc(num_elements * type_size);
    if (data_buffer == NULL) {
        fprintf(stderr, "Error allocating memory.\n");
        H5Sclose(dataspace);
        H5Tclose(datatype);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    printf("Memory allocated successfully.\n");

    // Read the data
    status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_buffer);
    if (status < 0) {
        fprintf(stderr, "Error reading dataset: %s\n", data->dataset_name);
        free(data_buffer);
        H5Sclose(dataspace);
        H5Tclose(datatype);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }
    printf("Data read successfully.\n");

    // Print the data based on the datatype
    if (H5Tequal(datatype, H5T_NATIVE_INT)) {
        int *int_data = (int *)data_buffer;
        for (hsize_t i = 0; i < (num_elements > 100 ? 100 : num_elements); i++) {
            printf("%d ", int_data[i]);
        }
    } else if (H5Tequal(datatype, H5T_NATIVE_FLOAT)) {
        float *float_data = (float *)data_buffer;
        for (hsize_t i = 0; i < (num_elements > 100 ? 100 : num_elements); i++) {
            printf("%f ", float_data[i]);
        }
    } else if (H5Tequal(datatype, H5T_NATIVE_DOUBLE)) {
        double *double_data = (double *)data_buffer;
        for (hsize_t i = 0; i < (num_elements > 100 ? 100 : num_elements); i++) {
            printf("%f ", double_data[i]);
        }
    } else {
        printf("Unsupported datatype.\n");
    }
    printf("\n");

    // Clean up
    free(data_buffer);
    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return NULL;
}

int main() {
    char *filename = "/scratch/sbotha/2024-hitfinder-data/real-data/pk7kev3_11_2768_data_000001.h5";
    char *dataset = "entry/data/data";

    thread_data_t thread_data;
    thread_data.filename = filename;
    thread_data.dataset_name = dataset;

    read_h5_dataset(&thread_data);

    return 0;
}
