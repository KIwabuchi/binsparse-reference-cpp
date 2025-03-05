#pragma once

#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>
#include <fstream>

auto start_time() {
  return std::chrono::high_resolution_clock::now();
}

// Return the elapsed time in seconds
auto elapsed_time_sec(const std::chrono::time_point<std::chrono::high_resolution_clock>& start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
}

// Runs a command using std::system and returns the output
std::string run_command(const std::string &cmd) {
  // std::cout << cmd << std::endl;

  const std::string tmp_file("/tmp/tmp_command_result");
  std::string command(cmd + " > " + tmp_file);
  const auto ret = std::system(command.c_str());
  if (ret != 0) {
    return std::string("Failed to execute: " + cmd);
  }

  std::ifstream ifs(tmp_file);
  if (!ifs.is_open()) {
    return std::string("Failed to open: " + tmp_file);
  }

  std::string buf;
  buf.assign((std::istreambuf_iterator<char>(ifs)),
             std::istreambuf_iterator<char>());

  // Remove the temporary file
  std::filesystem::remove(tmp_file);

  return buf;
}

// Returns the result of `du -d 0 -h dir_path`
std::string get_dir_usage(const std::string &dir_path) {
  return run_command("du -d 0 -h " + dir_path + " | head -n 1");
}