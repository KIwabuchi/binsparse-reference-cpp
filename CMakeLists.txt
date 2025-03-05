cmake_minimum_required(VERSION 3.20)
project(binsparse)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(include)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
target_link_libraries(binsparse INTERFACE ${HDF5_C_LIBRARIES} hdf5::hdf5_cpp)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Dependencies needed only for examples/test

  # Google test
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # fmt
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1)
  FetchContent_MakeAvailable(fmt)

  # Boost
  FetchContent_Declare(
    Boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.87.0/boost-1.87.0-cmake.tar.gz
  )
  set(BOOST_INCLUDE_LIBRARIES json unordered interprocess container property_tree uuid)
  set(BOOST_UUID_LINK_LIBATOMIC OFF)
  FetchContent_MakeAvailable(Boost)
  foreach(lib ${BOOST_INCLUDE_LIBRARIES})
      list(APPEND BOOST_COMPONENTS "Boost::${lib}")
  endforeach()

  # Threads
  find_package(Threads)

  # Metall
  FetchContent_Declare(
    Metall
    GIT_REPOSITORY https://github.com/LLNL/metall.git
    GIT_TAG v0.30
  )
  FetchContent_MakeAvailable(Metall)

  set(CMAKE_CXX_FLAGS "-O3 -march=native")

  add_subdirectory(examples)
  add_subdirectory(test)

endif()
