#include <iostream>
#include <H5Cpp.h>
#include <vector>
#include <cstring>

extern "C" {

struct DatasetInfo {
    unsigned int* data;
    int data_size;
    int dims[3];
    int ndims;
};

DatasetInfo* read_h5_dataset(const char* filename, const char* dataset_name) {
    DatasetInfo* info = new DatasetInfo;
    std::memset(info, 0, sizeof(DatasetInfo));

    try {
        // Open the HDF5 file
        H5::H5File file(filename, H5F_ACC_RDONLY);
        std::cout << "File opened successfully." << std::endl;

        // Open the dataset
        H5::DataSet dataset = file.openDataSet(dataset_name);
        std::cout << "Dataset opened successfully." << std::endl;

        // Get the dataspace of the dataset
        H5::DataSpace dataspace = dataset.getSpace();
        std::cout << "Dataspace obtained successfully." << std::endl;

        // Get the dimensions of the dataset
        hsize_t dims[3] = {1, 1, 1};
        info->ndims = dataspace.getSimpleExtentDims(dims, NULL);
        std::cout << "Number of dimensions: " << info->ndims << std::endl;
        std::cout << "Dimensions: [" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]" << std::endl;

        // Allocate buffer for the data
        info->data_size = dims[0] * dims[1] * dims[2];
        info->data = new unsigned int[info->data_size];
        std::cout << "Memory allocated successfully." << std::endl;

        // Read the data
        dataset.read(info->data, H5::PredType::NATIVE_UINT);
        std::cout << "Data read successfully." << std::endl;

        // Copy the dimensions
        for (int i = 0; i < info->ndims; ++i) {
            info->dims[i] = static_cast<int>(dims[i]);
        }
        if (info->ndims == 2) {
            info->dims[2] = 1;  // Set third dimension to 1 for 2D datasets
        }

    } catch (H5::FileIException &error) {
        std::cerr << "File error: " << error.getCDetailMsg() << std::endl;
        delete[] info->data;
        delete info;
        return NULL;
    } catch (H5::DataSetIException &error) {
        std::cerr << "Dataset error: " << error.getCDetailMsg() << std::endl;
        delete[] info->data;
        delete info;
        return NULL;
    } catch (H5::DataSpaceIException &error) {
        std::cerr << "Dataspace error: " << error.getCDetailMsg() << std::endl;
        delete[] info->data;
        delete info;
        return NULL;
    }

    return info;
}

void free_data(DatasetInfo* info) {
    delete[] info->data;
    delete info;
}

} // extern "C"
