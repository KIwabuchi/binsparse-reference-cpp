#include "util.hpp"
#include <binsparse/binsparse.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <gtest/gtest.h>

TEST(BinsparseReadWrite, CSRFormat) {
  using T = float;
  using I = std::size_t;
  using A = metall::manager::allocator_type<T>;
  using M = binsparse::__detail::csr_matrix_owning<T, I, A>;

  std::string binsparse_path = "out.bsp.metall";

  auto base_path = find_prefix(files.front());

  for (auto&& file : files) {
    // Keep some values for checking values after reading
    I nnz;
    I m;
    I n;
    std::vector<T> values;
    std::vector<I> colind;
    std::vector<I> row_ptr;

    {
      metall::manager manager(metall::create_only, binsparse_path);
      auto& x = *(manager.construct<M>("binsparse-matrix")
                                            (manager.get_allocator<>()));

      auto file_path = base_path + file;
      binsparse::__detail::mmread<T, I, M>(file_path,x);

      auto&& [num_rows, num_columns] = x.shape();
      binsparse::csr_matrix<T, I> matrix{x.values().data(), x.colind().data(),
                                         x.rowptr().data(), num_rows,
                                         num_columns,       I(x.size())};
      binsparse::write_csr_matrix(manager, matrix);

      nnz = matrix.nnz;
      m = matrix.m;
      n = matrix.n;
      for (I i = 0; i < nnz; i++) {
        values.push_back(matrix.values[i]);
      }
      for (I i = 0; i < nnz; i++) {
        colind.push_back(matrix.colind[i]);
      }
      for (I i = 0; i < m + 1; i++) {
        row_ptr.push_back(matrix.row_ptr[i]);
      }
    }

    {
      metall::manager manager(metall::open_only, binsparse_path);
      auto matrix_ = binsparse::read_csr_matrix<T, I>(manager);

      EXPECT_EQ(matrix_.nnz, nnz);
      EXPECT_EQ(matrix_.m, m);
      EXPECT_EQ(matrix_.n, n);

      for (I i = 0; i < matrix_.nnz; i++) {
        EXPECT_EQ(matrix_.values[i], values[i]);
      }

      for (I i = 0; i < matrix_.nnz; i++) {
        EXPECT_EQ(matrix_.colind[i], colind[i]);
      }

      for (I i = 0; i < matrix_.m + 1; i++) {
        EXPECT_EQ(matrix_.row_ptr[i], row_ptr[i]);
      }
    }
  }

  metall::manager::remove(binsparse_path);
}
