#include "util.hpp"
#include <binsparse/binsparse.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <gtest/gtest.h>

TEST(BinsparseReadWrite, CSRFormat) {
  using T = float;
  using I = std::size_t;

  std::string binsparse_dir = "out.bsp.metall";

  auto base_path = find_prefix(files.front());

  for (auto&& file : files) {
    metall::manager manager(metall::create_only, binsparse_dir);
    using A = metall::manager::allocator_type<T>;
    auto alloc = manager.get_allocator<A>();

    auto file_path = base_path + file;
    auto x = binsparse::__detail::mmread<
        T, I, binsparse::__detail::csr_matrix_owning<T, I, A>>(file_path,
                                                               alloc);

    auto&& [num_rows, num_columns] = x.shape();
    binsparse::csr_matrix<T, I> matrix{x.values().data(), x.colind().data(),
                                       x.rowptr().data(), num_rows,
                                       num_columns,       I(x.size())};
    binsparse::write_csr_matrix(manager, std::move(x));

    auto x_ = binsparse::read_csr_matrix<T, I, A>(manager);
    std::tie(num_rows, num_columns) = x_.shape();
    binsparse::csr_matrix<T, I> matrix_{x_.values().data(), x_.colind().data(),
                                        x_.rowptr().data(), num_rows,
                                        num_columns,        I(x_.size())};

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
  }

  metall::manager::remove(binsparse_dir);
}
