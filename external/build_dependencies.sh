#!/bin/bash

MATLABDIR=${MATLABDIR:-/usr/local/MATLAB/R2015b}

EXTERNAL_ROOT=$(pwd)


# Quit on error
set -e

########################################################################
#                             Build Caffe                              #
########################################################################
pushd caffe

mkdir -p build
pushd build

cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_ROOT}/caffe-bin \
    -DAtlas_CBLAS_LIBRARY=/usr/lib64/atlas/libtatlas.so \
    -DAtlas_BLAS_LIBRARY=/usr/lib64/atlas/libtatlas.so \
    -DAtlas_LAPACK_LIBRARY=/usr/lib64/atlas/libtatlas.so \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_HOST_COMPILER=/usr/bin/g++ \
    -DCUDA_NVCC_FLAGS="-Xcompiler -std=c++98" \
    -DBUILD_matlab=ON \
    -DMatlab_DIR=${MATLABDIR} \
    ..

make -j4
make install

popd
popd
