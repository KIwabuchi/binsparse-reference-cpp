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
  for (I i = 0; i < matrix.nnz; i++) {
    [[maybe_unused]] volatile const T value = matrix.values[i];
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

void bench_metall(const std::filesystem::path& binsparse_path) {
  auto start = start_time();
  metall::manager manager(metall::open_only, binsparse_path);
  auto matrix_ = binsparse::read_csr_matrix<T, I>(manager);
  auto elapsed = elapsed_time_sec(start);
  std::cout << "Reading binsparse matrix took " << elapsed << " s" << std::endl;

  start = start_time();
  touch_matrix(matrix_);
  elapsed = elapsed_time_sec(start);
  std::cout << "Touching matrix took " << elapsed << " s" << std::endl;
}

int main(int argc, char** argv) {
  std::filesystem::path binsparse_path = argv[1];
  std::string mode = argv[2];

  if (mode == "h5") {
    H5::DSetCreatPropList prop;
    bench_hdf5(binsparse_path);
  } else if (mode == "metall") {
    bench_metall(binsparse_path);
  } else {
    std::cerr << "Unknown mode" << std::endl;
    return 1;
  }

  return 0;
 }