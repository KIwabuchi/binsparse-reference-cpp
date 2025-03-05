# !/bin/bash

# This script runs the benchmark for the given matrix dataset and binsparse path
# Example usage:
# cd build
# make benchmark_write benchmark_read
# ../example/run_benchmark.sh -d /path/to/matrix/dataset -b /path/to/binsparse

MATRIX_DATASET=""
BINSPARSE_PATH=""

# Parse command line arguments
while getopts "d:b:" opt; do
  case ${opt} in
    d)
      MATRIX_DATASET=${OPTARG}
      ;;
    b)
      BINSPARSE_PATH=${OPTARG}
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      ;;
  esac
done

drop_caches() {
  echo "Dropping caches"

  sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

  # On LC
  # srun --drop-caches true
}

run_benchmark() {
  mode=$1
  dataset=$2
  binsparse_path="${3}.${mode}"
  do_drop_caches=$4 # Drop caches before reading benchmark

  echo "Running benchmark for ${dataset} with mode ${mode} and binsparse path ${binsparse_path}"

  # Check if the binsparse path exists
  if [ -d ${binsparse_path} ]; then
    echo "Warning: the binsparse path ${binsparse_path} already exist"
    echo "Do you want to remove the existing binsparse path? (y/n)"
    read answer
    if [ ${answer} = "y" ]; then
      rm -rf ${binsparse_path}
    else
      echo "Exiting the script"
      exit 1
    fi
  fi

  drop_caches

  ./examples/benchmark_write ${dataset} ${binsparse_path} ${mode}

  if [ ${do_drop_caches} -eq 1 ]; then
    drop_caches
  fi
  ./examples/benchmark_read ${binsparse_path} ${mode}

  echo "Removing the binsparse path ${binsparse_path}"
  rm -rf ${binsparse_path}

  echo ""
}

run_benchmark "h5" ${MATRIX_DATASET} ${BINSPARSE_PATH} 1
run_benchmark "metall" ${MATRIX_DATASET} ${BINSPARSE_PATH} 1
run_benchmark "h5" ${MATRIX_DATASET} ${BINSPARSE_PATH} 0
run_benchmark "metall" ${MATRIX_DATASET} ${BINSPARSE_PATH} 0