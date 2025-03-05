#include "H5Cpp.h"
#include <vector>
#include <iostream>

#include "benchmark_util.hpp"

using value_type = float;

int main() {
  const hsize_t file_size = 1ULL << 30;
  const hsize_t num_elements = file_size / sizeof(value_type);

  const H5std_string FILE_NAME("/dev/shm/large_array.h5");
  const H5std_string DATASET_NAME("big_int_array");

  try {
    // Create a vector with dummy data
    std::vector<value_type> data(num_elements, 1.23);

    // Create HDF5 file
    H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);

    // Define dataspace with 1D array of size `num_elements`
    hsize_t dims[1] = { num_elements };
    H5::DataSpace dataspace(1, dims);

    // Create dataset of native value_type type
    H5::DataSet dataset = file.createDataSet(
        DATASET_NAME,
        H5::PredType::NATIVE_FLOAT,
        dataspace
    );

    // Write the data to the dataset
    auto start = start_time();
    dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
    dataset.close();
    dataspace.close();
    auto elapsed = elapsed_time_sec(start);

    std::cout << "Successfully wrote " << num_elements << " to " << FILE_NAME << std::endl;
    std::cout << "Writing took " << elapsed << " s" << std::endl;
  }
  catch (H5::FileIException& e) {
    e.printErrorStack();
    return 1;
  }
  catch (H5::DataSetIException& e) {
    e.printErrorStack();
    return 1;
  }
  catch (H5::DataSpaceIException& e) {
    e.printErrorStack();
    return 1;
  }

  return 0;
}
