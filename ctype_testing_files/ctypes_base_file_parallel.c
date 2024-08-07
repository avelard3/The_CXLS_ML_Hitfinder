#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <mpi.h>

// Global variables
hid_t file_id, dataset_id, space_id;
hsize_t dims[3];
int ndims;
unsigned int* data = NULL;
MPI_Comm comm = MPI_COMM_WORLD;
MPI_Info info = MPI_INFO_NULL;

int open_dataset(const char* filename, const char* dataset_name, int* out_dims) {
    // Initialize MPI if not already initialized
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(NULL, NULL);
    }

    // Initialize MPI variables
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    // Set up file access property list with parallel I/O access
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);

    // Open the file with parallel I/O
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);
    if (file_id < 0) {
        if (mpi_rank == 0) fprintf(stderr, "Could not open file: %s\n", filename);
        return -1;
    }

    // Open the dataset
    dataset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        if (mpi_rank == 0) fprintf(stderr, "Could not open dataset: %s\n", dataset_name);
        H5Fclose(file_id);
        return -1;
    }

    // Get the dataspace
    space_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims != 3) {
        if (mpi_rank == 0) fprintf(stderr, "Unexpected number of dimensions: %d\n", ndims);
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
    // Initialize MPI variables
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    if (layer_index < 0 || layer_index >= dims[0]) {
        if (mpi_rank == 0) fprintf(stderr, "Invalid layer index: %d\n", layer_index);
        return NULL;
    }

    hsize_t offset[3] = {layer_index, 0, 0};
    hsize_t count[3] = {1, dims[1] / mpi_size, dims[2]};
    hsize_t mem_offset[3] = {0, mpi_rank * count[1], 0};
    hsize_t mem_count[3] = {1, count[1], dims[2]};

    // Allocate memory for the layer portion
    *layer_size = count[1] * dims[2];
    data = (unsigned int*) malloc(*layer_size * sizeof(unsigned int));
    if (data == NULL) {
        if (mpi_rank == 0) fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Select hyperslab in the file
    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Create memory dataspace
    hid_t mem_space_id = H5Screate_simple(3, mem_count, NULL);

    // Select hyperslab in the memory space
    H5Sselect_hyperslab(mem_space_id, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

    // Set up collective transfer properties list
    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    // Read the data
    if (H5Dread(dataset_id, H5T_NATIVE_UINT, mem_space_id, space_id, xfer_plist_id, data) < 0) {
        if (mpi_rank == 0) fprintf(stderr, "Could not read data\n");
        free(data);
        data = NULL;
    }

    H5Sclose(mem_space_id);
    H5Pclose(xfer_plist_id);
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
    MPI_Finalize();
}
