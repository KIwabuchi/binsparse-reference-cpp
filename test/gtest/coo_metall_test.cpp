#include "util.hpp"
#include <binsparse/binsparse.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <gtest/gtest.h>

TEST(BinsparseReadWrite, COOFormat) {
  using T = float;
  using I = std::size_t;
  using A = metall::manager::allocator_type<T>;
  using M = binsparse::__detail::coo_matrix_owning<T, I, A>;

  std::string binsparse_dir = "out.bsp.metall";

  auto base_path = find_prefix(files.front());

  for (auto&& file : files) {
    // Keep some values for checking values afeter reading
    I nnz;
    I m;
    I n;
    std::vector<T> values;
    std::vector<I> rowind;
    std::vector<I> colind;
    {
      metall::manager manager(metall::create_only, binsparse_dir);
      auto& x = *(manager.construct<M>("binsparse-matrix")
                                            (manager.get_allocator<>()));

      auto file_path = base_path + file;
      binsparse::__detail::mmread<T, I, M>(file_path,x);

      auto&& [num_rows, num_columns] = x.shape();
      binsparse::coo_matrix<T, I> matrix{x.values().data(), x.rowind().data(),
                                         x.colind().data(), num_rows,
                                         num_columns,       I(x.size())};
      binsparse::write_coo_matrix(manager, matrix);

      nnz = matrix.nnz;
      m = matrix.m;
      n = matrix.n;
      for (I i = 0; i < nnz; i++) {
        values.push_back(matrix.values[i]);
      }
      for (I i = 0; i < nnz; i++) {
        rowind.push_back(matrix.rowind[i]);
      }
      for (I i = 0; i < nnz; i++) {
        colind.push_back(matrix.colind[i]);
      }
    }

    {
      metall::manager manager(metall::open_only, binsparse_dir);
      auto matrix_ = binsparse::read_coo_matrix<T, I>(manager);

      EXPECT_EQ(matrix_.nnz, nnz);
      EXPECT_EQ(matrix_.m, m);
      EXPECT_EQ(matrix_.n, n);

      for (I i = 0; i < matrix_.nnz; i++) {
        EXPECT_EQ(matrix_.values[i], values[i]);
      }

      for (I i = 0; i < matrix_.nnz; i++) {
        EXPECT_EQ(matrix_.rowind[i], rowind[i]);
      }

      for (I i = 0; i < matrix_.nnz; i++) {
        EXPECT_EQ(matrix_.colind[i], colind[i]);
      }
    }
  }

  metall::manager::remove(binsparse_dir);
}
