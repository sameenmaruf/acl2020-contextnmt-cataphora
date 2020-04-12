This is a DyNet implementation of Tranformer-HAN-encoder (https://www.aclweb.org/anthology/D18-1325/) used in our ACL'20 paper:

KayYen Wong, Sameen Maruf and Gholamreza Haffari: Contextual Neural Machine Translation Improves Translation of Cataphoric Pronouns. 

Please cite our paper if you use this implementation. 

# Dependencies

Before compiling dynet, you need:

 * [Eigen](https://bitbucket.org/eigen/eigen), using the development version (not release), e.g. 3.3.beta2 (http://bitbucket.org/eigen/eigen/get/3.3-beta2.tar.bz2)

 * [cuda](https://developer.nvidia.com/cuda-toolkit) version 7.5 or higher

 * [boost](http://www.boost.org/), e.g., 1.58 using *libboost-all-dev* ubuntu package

 * [cmake](https://cmake.org/), e.g., 3.5.1 using *cmake* ubuntu package

# Building

First, clone the repository

git clone https://github.com/sameenmaruf/acl2020-contextnmt-cataphora.git

As mentioned above, you'll need the latest [development] version of eigen

hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb

A version of latest DyNet (v2.0.3) is already included (e.g., dynet folder). 

# CPU build

Compiling to execute on a CPU is as follows

    mkdir build_cpu
    cd build_cpu
    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH [-DBoost_NO_BOOST_CMAKE=ON]
    make -j 2

Boost note. The "-DBoost_NO_BOOST_CMAKE=ON" can be optional but if you have a trouble of boost-related build error(s), adding it will help to overcome. 

    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DBoost_NO_BOOST_CMAKE=TRUE -DENABLE_BOOST=TRUE

substituting in your EIGEN_PATH. 

This will build the following binaries
    
    build_cpu/transformer-train
    build_cpu/transformer-decode
    build_cpu/transformer-computerep
    build_cpu/transformer-context
    build_cpu/transformer-single-context-decode
    build_cpu/transformer-context-decode
    
# GPU build

Building on the GPU uses the Nvidia CUDA library, currently tested against version 8.0.61.
The process is as follows

    mkdir build_gpu
    cd build_gpu
    cmake .. -DBACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=CUDA_PATH -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE -DENABLE_BOOST=TRUE
    make -j 2

substituting in your EIGEN_PATH and CUDA_PATH folders, as appropriate. 

This will result in the binaries

    build_gpu/transformer-train
    build_gpu/transformer-decode
    build_gpu/transformer-computerep
    build_gpu/transformer-context
    build_gpu/transformer-single-context-decode
    build_gpu/transformer-context-decode

# Using the model

See readme_commands.txt

# References

The original Transformer implementation used in our code is available at https://github.com/duyvuleo/Transformer-DyNet

## Contacts

Sameen Maruf (sameen.maruf@monash.edu; sameen.maruf@gmail.com)

---
Updated April 2020
