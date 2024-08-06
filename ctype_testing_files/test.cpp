#include <iostream>
#include <H5Cpp.h>
#include <vector>

int main() {
    const std::string filename = "/scratch/sbotha/2024-hitfinder-data/real-data/pk7kev3_11_2768_data_000001.h5";
    const std::string datasetName = "entry/data/data";

    try {
        // Open the HDF5 file
        H5::H5File file(filename, H5F_ACC_RDONLY);
        std::cout << "File opened successfully." << std::endl;

        // Open the dataset
        H5::DataSet dataset = file.openDataSet(datasetName);
        std::cout << "Dataset opened successfully." << std::endl;

        // Get the dataspace of the dataset
        H5::DataSpace dataspace = dataset.getSpace();
        std::cout << "Dataspace obtained successfully." << std::endl;

        // Get the dimensions of the dataset
        hsize_t dims[3];
        int ndims = dataspace.getSimpleExtentDims(dims, NULL);
        std::cout << "Number of dimensions: " << ndims << std::endl;
        std::cout << "Dimensions: [" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]" << std::endl;

        // Allocate buffer for the data
        std::vector<int> dataBuffer(dims[0] * dims[1] * dims[2]);
        std::cout << "Memory allocated successfully." << std::endl;

        // Read the data
        dataset.read(dataBuffer.data(), H5::PredType::NATIVE_INT);
        std::cout << "Data read successfully." << std::endl;

        // Print each slice (image)
        // for (size_t c = 0; c < dims[0]; ++c) {
        //     std::cout << "Slice " << c << ":" << std::endl;
        //     for (size_t i = 0; i < dims[1]; ++i) {
        //         for (size_t j = 0; j < dims[2]; ++j) {
        //             std::cout << dataBuffer[c * dims[1] * dims[2] + i * dims[2] + j] << " ";
        //         }
        //         std::cout << std::endl;
        //     }
        //     std::cout << std::endl; // Extra newline for separation between slices
        // }

    } catch (H5::FileIException &error) {
        std::cerr << "File error: " << error.getCDetailMsg() << std::endl;
        return -1;
    } catch (H5::DataSetIException &error) {
        std::cerr << "Dataset error: " << error.getCDetailMsg() << std::endl;
        return -1;
    } catch (H5::DataSpaceIException &error) {
        std::cerr << "Dataspace error: " << error.getCDetailMsg() << std::endl;
        return -1;
    }

    return 0;
}
