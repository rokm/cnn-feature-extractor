#!/bin/bash

# Matlab directory; set only if not already set
MATLABDIR=${MATLABDIR:-/usr/local/MATLAB/R2016b}

# CUDA host compiler
CUDA_HOST_COMPILER=${HOST_COMPILER:-/usr/bin/g++}

# Get the project's root directory (i.e., the location of this script)
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Quit on error
set -e

########################################################################
#                             Build Caffe                              #
########################################################################
# Fedora dependencies:
#  cmake gcc-c++ boost-devel glog-devel gflags-devel  protobuf-devel
#  hdf5-devel lmdb-devel leveldb-devel snappy-devel openblas-devel
#  python-devel python2-numpy
#  cuda-devel cudnn-devel
#
# Ubuntu dependencies:
#  cmake build-essential libboost-all-dev libgflags-dev libgoogle-glog-dev
#  libprotobuf-dev protobuf-compiler libhdf5-dev liblmdb-dev libleveldb-dev
#  libsnappy-dev libopenblas-dev python-dev python-numpy
#  nvidia-cuda-toolkit
CAFFE_SOURCE_DIR="${ROOT_DIR}/external/caffe"
CAFFE_BUILD_DIR="${CAFFE_SOURCE_DIR}/build"
CAFFE_INSTALL_DIR="${ROOT_DIR}/external/caffe-bin"

# Make sure the submodule has been checked out
if [ ! -f "${CAFFE_SOURCE_DIR}/.git" ]; then
    echo "The caffe submodule does not appear to be checked out!"
    exit 1
fi

# Build and install
mkdir -p "${CAFFE_BUILD_DIR}"

cmake \
    -H"${CAFFE_SOURCE_DIR}" \
    -B"${CAFFE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="${CAFFE_INSTALL_DIR}" \
    -DBLAS=open \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_HOST_COMPILER="${CUDA_HOST_COMPILER}" \
    -DCUDA_NVCC_FLAGS="-Xcompiler -std=c++98" \
    -DUSE_OPENCV=OFF \
    -DBUILD_python=OFF \
    -DBUILD_matlab=ON \
    -DMatlab_DIR=${MATLABDIR}

make -j4 -C "${CAFFE_BUILD_DIR}"
make install -C "${CAFFE_BUILD_DIR}"
