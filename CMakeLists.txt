cmake_minimum_required(VERSION 3.26)
project(torch_gds)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

# --- CUDA (BEGIN) ---
# Specify host compiler used for NVCC.
# Or add the following line to `~/.bash_profile` :
#   export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc-12.2'
set(CUDA_NVCC_FLAGS -ccbin=/usr/bin/gcc-12.2)
set(TCNN_CUDA_ARCHITECTURES 75) # Turing
# --- CUDA (END) ---

# --- LibTorch (BEGIN) ---
set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# --- LibTorch (END) ---


add_executable(torch_gds
        embedding.cpp)

target_link_libraries(torch_gds "${TORCH_LIBRARIES}")