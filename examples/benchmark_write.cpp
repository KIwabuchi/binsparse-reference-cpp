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

void bench_hdf5(const std::filesystem::path& file_path,
                const std::filesystem::path& binsparse_path) {

  using M = binsparse::__detail::csr_matrix_owning<T, I>;

  M x;
  {
    auto start = start_time();
    binsparse::__detail::mmread<T, I, M>(file_path, x);
    auto elapsed = stop_time(start);
    std::cout << "Reading took " << elapsed << " ms" << std::endl;
   }

  auto&& [num_rows, num_columns] = x.shape();
  binsparse::csr_matrix<T, I> matrix{x.values().data(), x.colind().data(),
                                     x.rowptr().data(), num_rows,
                                     num_columns,       I(x.size())};

  {
    const int def_level = 0; // No compression for performance comparison
    auto start = start_time();
    binsparse::write_csr_matrix(binsparse_path, matrix, {}, def_level);
    auto elapsed = stop_time(start);
    std::cout << "Writing took " << elapsed << " ms" << std::endl;
   }
}

void bench_metall(const std::filesystem::path& file_path,
                  const std::filesystem::path& binsparse_path) {

  using A = metall::manager::allocator_type<T>;
  using M = binsparse::__detail::csr_matrix_owning<T, I, A>;

  auto* manager = new metall::manager(metall::create_only, binsparse_path);
  auto& x = *(manager->construct<M>(metall::unique_instance)
                                      (manager->get_allocator<>()));

  {
    auto start = start_time();
    binsparse::__detail::mmread<T, I, M>(file_path, x);
    auto elapsed = stop_time(start);
    std::cout << "Reading original matrix took " << elapsed << " ms" << std::endl;
  }

  auto&& [num_rows, num_columns] = x.shape();
  binsparse::csr_matrix<T, I> matrix{x.values().data(), x.colind().data(),
                             x.rowptr().data(), num_rows,
                             num_columns,       I(x.size())};

  {
    auto start = start_time();
    binsparse::write_csr_matrix(*manager, matrix);
    // manager->flush();
    delete manager;
    auto elapsed = stop_time(start);
    std::cout << "Writing binsparse matrix took " << elapsed << " ms" << std::endl;
  }
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