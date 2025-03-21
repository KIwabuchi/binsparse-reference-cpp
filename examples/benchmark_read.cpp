#include <iostream>
#include <filesystem>
#include <chrono>

#include <binsparse/binsparse.hpp>
#include <metall/metall.hpp>


auto start_time() {
  return std::chrono::high_resolution_clock::now();
}

// Return the elapsed time in milliseconds
auto stop_time(const std::chrono::time_point<std::chrono::high_resolution_clock>& start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
}

using T = float;
using I = std::size_t;

void touch_matrix(const binsparse::csr_matrix<T, I>& matrix) {
  for (I i = 0; i < matrix.nnz; i++) {
    [[maybe_unused]] volatile const T value = matrix.values[i];
  }

  for (I i = 0; i < matrix.nnz; i++) {
    [[maybe_unused]] volatile const T value = matrix.values[i];
  }
}

void bench_hdf5(const std::filesystem::path& binsparse_file) {
  auto start = start_time();
  auto matrix_ = binsparse::read_csr_matrix<T, I>(binsparse_file);
  auto elapsed = stop_time(start);
  std::cout << "Reading binsparse matrix took " << elapsed << " ms" << std::endl;

  start = start_time();
  touch_matrix(matrix_);
  elapsed = stop_time(start);
  std::cout << "Touching matrix took " << elapsed << " ms" << std::endl;
}

void bench_metall(const std::filesystem::path& binsparse_dir) {

  auto start = start_time();
  metall::manager manager(metall::open_only, binsparse_dir);
  auto matrix_ = binsparse::read_csr_matrix<T, I>(manager);
  auto elapsed = stop_time(start);
  std::cout << "Reading binsparse matrix took " << elapsed << " ms" << std::endl;

  start = start_time();
  touch_matrix(matrix_);
  elapsed = stop_time(start);
  std::cout << "Touching matrix took " << elapsed << " ms" << std::endl;
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