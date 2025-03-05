/// Reads a .mtx file and writes it to a binsparse matrix in HDF5 or Metall format.

#include <iostream>
#include <filesystem>

#include <binsparse/binsparse.hpp>
#include <metall/metall.hpp>

#include "benchmark_util.hpp"

using T = float;
using I = std::size_t;

void bench_hdf5(const std::filesystem::path& file_path,
                const std::filesystem::path& binsparse_path) {
  using M = binsparse::__detail::csr_matrix_owning<T, I>;

  M x;
  {
    auto start = start_time();
    binsparse::__detail::mmread<T, I, M>(file_path, x);
    auto elapsed = elapsed_time_sec(start);
    std::cout << "Reading original matrix took " << elapsed << " s" << std::endl;
   }

  auto&& [num_rows, num_columns] = x.shape();
  binsparse::csr_matrix<T, I> matrix{x.values().data(), x.colind().data(),
                                     x.rowptr().data(), num_rows,
                                     num_columns,       I(x.size())};

  {
    const int def_level = 0; // No compression for performance comparison
    auto start = start_time();
    binsparse::write_csr_matrix(binsparse_path, matrix, {}, def_level);
    auto elapsed = elapsed_time_sec(start);
    std::cout << "Writing HDF5 binsparse matrix took " << elapsed << " s" << std::endl;
   }

  std::cout << "HDF5 file size: \n" << get_dir_usage(binsparse_path);
}

void bench_metall(const std::filesystem::path& file_path,
                  const std::filesystem::path& binsparse_path) {
  using A = metall::manager::allocator_type<T>;
  using M = binsparse::__detail::csr_matrix_owning<T, I, A>;

  // Create a Metall manager instance
  auto* manager = new metall::manager(metall::create_only, binsparse_path);
  // Allocate and construct an instance of M using Metall.
  // manager::construct() returns a pointer to the instance.
  // From now on, 'x' can be used as a normal instance of M.
  auto& x = *(manager->construct<M>("binsparse-matrix")
                                      (manager->get_allocator<>()));

  {
    auto start = start_time();
    binsparse::__detail::mmread<T, I, M>(file_path, x);
    auto elapsed = elapsed_time_sec(start);
    std::cout << "Reading original matrix took " << elapsed << " s" << std::endl;
  }

  auto&& [num_rows, num_columns] = x.shape();
  binsparse::csr_matrix<T, I> matrix{x.values().data(), x.colind().data(),
                             x.rowptr().data(), num_rows,
                             num_columns,       I(x.size())};

  {
    auto start = start_time();
    binsparse::write_csr_matrix(*manager, matrix);
    // Metall manager's destructor make sure to flush the data to the storage
    // and close the Metall datastore directory (i.e., binsparse_path).
    // This process does not destructor the instance of M.
    // We delete manager here to do a fair comparison with the HDF5 case as
    // this process is somewhat similar to closing the HDF5 file.
    delete manager;
    auto elapsed = elapsed_time_sec(start);
    std::cout << "Writing Metall binsparse matrix took " << elapsed << " s" << std::endl;
  }

  std::cout << "Metall directory size: \n" << get_dir_usage(binsparse_path);
}

int main(int argc, char** argv) {
  std::filesystem::path file_path = argv[1];
  std::filesystem::path binsparse_path = argv[2];
  std::string mode = argv[3];

  if (mode == "h5") {
    H5::DSetCreatPropList prop;
    bench_hdf5(file_path, binsparse_path);
  } else if (mode == "metall") {
    bench_metall(file_path, binsparse_path);
  } else {
    std::cerr << "Unknown mode" << std::endl;
    return 1;
  }

  return 0;
 }