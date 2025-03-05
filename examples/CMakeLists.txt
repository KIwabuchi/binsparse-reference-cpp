function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} binsparse fmt Metall::Metall Threads::Threads ${BOOST_COMPONENTS})
endfunction()

add_example(convert_binsparse)
add_example(text2hdf5)
add_example(inspect_binsparse)
add_example(convert_matrixmarket)
add_example(benchmark_write)
add_example(benchmark_read)

add_example(benchmark_hdf5_simple)