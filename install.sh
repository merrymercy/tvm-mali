sudo apt-get update

sudo apt install llvm-4.0
sudo apt install scons
sudo apt install libopenblas-dev
sudo apt-get -y install git cmake build-essential g++-4.8 c++-4.8 liblapack* libblas* libopencv*

git clone --recursive https://github.com/dmlc/nnvm.git
git clone https://github.com/ARM-software/ComputeLibrary.git --branch v17.12
git clone --recursive https://github.com/apache/incubator-mxnet.git 

# build nnvm/tvm
cd nnvm/tvm
make USE_OPENCL=1  LLVM_CONFIG=llvm-config-4.0 -j4
cd ..
make
cd ..

# build arm compute library
cd ComputeLibrary
scons Werror=1 neon=1 opencl=1 examples=1 os=linux arch=arm64-v8a embed_kernels=1 build=native -j4
cp ../acl_test.cc .

g++ acl_test.cc build/utils/*.o -O2 -std=c++11 -I. -Iinclude -Lbuild -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o acl_test
cp acl_test ..
cd ..

# build mxnet
cd incubator-mxnet
make -j2 USE_OPENCV=0 USE_BLAS=openblas
cd ..

