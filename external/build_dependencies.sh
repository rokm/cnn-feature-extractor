#!/bin/sh

MATLABDIR=/usr/local/MATLAB/R2015b

EXTERNAL_ROOT=$(pwd)


########################################################################
#                             Build Caffe                              #
########################################################################
pushd caffe

mkdir build
pushd build

cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_ROOT}/caffe-bin \
    -DAtlas_CBLAS_LIBRARY=/usr/lib64/atlas/libtatlas.so \
    -DAtlas_BLAS_LIBRARY=/usr/lib64/atlas/libtatlas.so \
    -DAtlas_LAPACK_LIBRARY=/usr/lib64/atlas/libtatlas.so \
    -DBUILD_matlab=ON \
    -DMatlab_DIR=${MATLABDIR} \
    ..

make -j4
make install

popd
popd
