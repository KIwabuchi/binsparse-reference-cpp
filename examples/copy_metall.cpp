#include <iostream>
#include <filesystem>

#include <metall/metall.hpp>

#include "benchmark_util.hpp"

int main(int argc, char** argv) {
  const std::filesystem::path from = argv[1];
  const std::filesystem::path to = argv[2];

  auto start_copy = start_time();
  metall::manager::copy(from, to);
  auto elapsed_copy = elapsed_time_sec(start_copy);
  std::cout << "Copying " << from << " -> " << to << " took " << elapsed_copy << " s)" << std::endl;

  return 0;
}