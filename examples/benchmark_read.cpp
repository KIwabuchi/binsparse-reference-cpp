/// Read a binsparse matrix from a file and touch all its values.
/// Use benchmark_write.cpp to generate the binsparse file.

#include <iostream>
#include <filesystem>

#include <binsparse/binsparse.hpp>
#include <metall/metall.hpp>

#include "benchmark_util.hpp"

using T = float;
using I = std::size_t;

// TODO: Implement a more realistic access pattern
void touch_matrix(const binsparse::csr_matrix<T, I>& matrix) {
  #pragma omp parallel for
  for (I i = 0; i < matrix.nnz; i++) {
    [[maybe_unused]] volatile const T value = matrix.values[i];
    if (i < 10) { // Print the first 10 values for validation
      std::cout << "Value at index " << i << ": " << value << std::endl;
    }
  }
}

void bench_hdf5(const std::filesystem::path& binsparse_path) {
  auto start = start_time();
  auto matrix_ = binsparse::read_csr_matrix<T, I>(binsparse_path);
  auto elapsed = elapsed_time_sec(start);
  std::cout << "Reading binsparse matrix took " << elapsed << " s" << std::endl;

  start = start_time();
  touch_matrix(matrix_);
  elapsed = elapsed_time_sec(start);
  std::cout << "Touching matrix took " << elapsed << " s" << std::endl;
}

void bench_metall(const std::filesystem::path& binsparse_path,
                  const bool metall_use_scratchpad) {
  const auto metall_path = metall_use_scratchpad ? "/dev/shm/metall-binsparse" : binsparse_path;

  auto start = start_time();
  if (metall_use_scratchpad) {
    std::cout << "Using scratchpad mode" << std::endl;
    auto start_copy = start_time();
    metall::manager::copy(binsparse_path, metall_path);
    auto elapsed_copy = elapsed_time_sec(start_copy);
    std::cout << "(copying binsparse matrix took " << elapsed_copy << " s)" << std::endl;
  }
  metall::manager manager(metall::open_only, metall_path);
  auto matrix_ = binsparse::read_csr_matrix<T, I>(manager);
  auto elapsed = elapsed_time_sec(start);
  std::cout << "Reading binsparse matrix took " << elapsed << " s" << std::endl;

  start = start_time();
  touch_matrix(matrix_);
  elapsed = elapsed_time_sec(start);
  std::cout << "Touching matrix took " << elapsed << " s" << std::endl;
}

int main(int argc, char** argv) {
  const std::filesystem::path binsparse_path = argv[1];
  const std::string mode = argv[2];
  bool metall_use_scratchpad = false;
  if (argc > 3) {
    metall_use_scratchpad = std::stoi(argv[3]) == 1;
  }

  if (mode == "h5") {
    H5::DSetCreatPropList prop;
    bench_hdf5(binsparse_path);
  } else if (mode == "metall") {
    bench_metall(binsparse_path, metall_use_scratchpad);
  } else {
    std::cerr << "Unknown mode" << std::endl;
    return 1;
  }

  return 0;
 }