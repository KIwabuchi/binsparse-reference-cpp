#include "util.hpp"
#include <binsparse/binsparse.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <gtest/gtest.h>

TEST(BinsparseReadWrite, CSRFormat) {
  using T = float;
  using I = std::size_t;
  using M = binsparse::__detail::csr_matrix_owning<T, I>;
  std::string binsparse_file = "out.bsp.hdf5";

  auto base_path = find_prefix(files.front());

  for (auto&& file : files) {
    auto file_path = base_path + file;
    M x;
    binsparse::__detail::mmread<T, I, M>(file_path, x);

    auto&& [num_rows, num_columns] = x.shape();
    binsparse::csr_matrix<T, I> matrix{x.values().data(), x.colind().data(),
                                       x.rowptr().data(), num_rows,
                                       num_columns,       I(x.size())};
    binsparse::write_csr_matrix(binsparse_file, matrix);

    auto matrix_ = binsparse::read_csr_matrix<T, I>(binsparse_file);

    EXPECT_EQ(matrix.nnz, matrix_.nnz);
    EXPECT_EQ(matrix.m, matrix_.m);
    EXPECT_EQ(matrix.n, matrix_.n);

    for (I i = 0; i < matrix.nnz; i++) {
      EXPECT_EQ(matrix.values[i], matrix_.values[i]);
    }

    for (I i = 0; i < matrix.nnz; i++) {
      EXPECT_EQ(matrix.colind[i], matrix_.colind[i]);
    }

    for (I i = 0; i < matrix.m + 1; i++) {
      EXPECT_EQ(matrix.row_ptr[i], matrix_.row_ptr[i]);
    }

    delete matrix_.values;
    delete matrix_.row_ptr;
    delete matrix_.colind;
  }

  std::filesystem::remove(binsparse_file);
}
