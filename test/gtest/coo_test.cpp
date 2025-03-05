#include "util.hpp"
#include <binsparse/binsparse.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <gtest/gtest.h>

TEST(BinsparseReadWrite, COOFormat) {
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
        T, I, binsparse::__detail::coo_matrix_owning<T, I, A>>(file_path,
                                                               alloc);

    auto&& [num_rows, num_columns] = x.shape();
    binsparse::coo_matrix<T, I> matrix{x.values().data(), x.rowind().data(),
                                       x.colind().data(), num_rows,
                                       num_columns,       I(x.size())};
    binsparse::write_coo_matrix(manager, std::move(x));

    auto x_ = binsparse::read_coo_matrix<T, I, A>(manager);
    std::tie(num_rows, num_columns) = x_.shape();
    binsparse::coo_matrix<T, I> matrix_{x_.values().data(), x_.rowind().data(),
                                        x_.colind().data(), num_rows,
                                        num_columns,        I(x_.size())};
    EXPECT_EQ(matrix.nnz, matrix_.nnz);
    EXPECT_EQ(matrix.m, matrix_.m);
    EXPECT_EQ(matrix.n, matrix_.n);

    for (I i = 0; i < matrix.nnz; i++) {
      EXPECT_EQ(matrix.values[i], matrix_.values[i]);
    }

    for (I i = 0; i < matrix.nnz; i++) {
      EXPECT_EQ(matrix.rowind[i], matrix_.rowind[i]);
    }

    for (I i = 0; i < matrix.nnz; i++) {
      EXPECT_EQ(matrix.colind[i], matrix_.colind[i]);
    }
  }

  metall::manager::remove(binsparse_dir);
}
