cmake_minimum_required(VERSION 3.30)

# Enable CMP0104 to ensure CUDA architectures are set properly
cmake_policy(SET CMP0104 NEW)

project(GaussFilter LANGUAGES C CXX CUDA)

set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

# Declare the FetchContent
FetchContent_Declare(
        CudaDependencies
        GIT_REPOSITORY https://github.com/NVIDIA/cuda-samples.git
        GIT_TAG master
)

# Make it available (this handles the population)
FetchContent_MakeAvailable(CudaDependencies)

# Include directories
include_directories(${FETCHCONTENT_BASE_DIR}/cudadependencies-src/Common)

# Set CUDA standard version and architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80)

# Set the CUDA root directory (adjust if needed)
set(CUDA_ROOT /usr/local/cuda)


# Include directories
include_directories(${CUDA_ROOT}/include)


# Create the executable
add_executable(${PROJECT_NAME} gauss_filter.cu)

## Link CUDA libraries directly
target_link_libraries(${PROJECT_NAME}
        freeimage
        ${CUDA_ROOT}/lib64/libcudart.so
        ${CUDA_ROOT}/lib64/libnppc.so
        ${CUDA_ROOT}/lib64/libnppial.so
        ${CUDA_ROOT}/lib64/libnppif.so
        ${CUDA_ROOT}/lib64/libnppicc.so
        ${CUDA_ROOT}/lib64/libnppig.so
        ${CUDA_ROOT}/lib64/libnppisu.so)


# Suppress deprecated GPU target warnings
add_compile_options(-Wno-deprecated-gpu-targets)
