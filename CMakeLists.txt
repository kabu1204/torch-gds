cmake_minimum_required(VERSION 3.0)
project(torch_gds)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(.)

# --- Threads ---
find_package(Threads REQUIRED)
# --- Threads ---

# --- Intel/PCM (BEGIN) ---
add_subdirectory(pcm)
# --- Intel/PCM (END) ---

# --- PCIeMonitor (BEGIN) ---
add_subdirectory(pcie)
# --- PCIeMonitor (END) ---

# --- PCIe monitor (BEGIN) ---
include_directories(${PROJECT_SOURCE_DIR})
# --- PCIe monitor (END) ---

# --- CUDA (BEGIN) ---
# Specify host compiler used for NVCC.
# Or add the following line to `~/.bash_profile` :
#   export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc-12.2'
# set(CUDA_NVCC_FLAGS -ccbin=/usr/bin/gcc-12.2)
set(TCNN_CUDA_ARCHITECTURES 89) # ADA_LOVELACE
# --- CUDA (END) ---


# --- LibTorch (BEGIN) ---
# set(Torch_DIR /home/ycy/repo/libtorch/share/cmake/Torch)
set(Torch_DIR /home/ycy/repo/pytorch/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# --- LibTorch (END) ---

add_subdirectory(torch_ext)

add_executable(torch_gds
        embedding.cpp
        )

target_link_libraries(torch_gds "${TORCH_LIBRARIES}" pcie_monitor Threads::Threads mycopy)