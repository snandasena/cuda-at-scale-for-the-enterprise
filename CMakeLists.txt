cmake_minimum_required(VERSION 3.30)

# Set CMake policy CMP0104 to NEW to ensure CUDA architectures are set properly
# CMP0104 allows defining CUDA_ARCHITECTURES in CMake 3.18+.
cmake_policy(SET CMP0104 NEW)

# Project details
project(GaussFilter LANGUAGES C CXX CUDA)

# Set CMake variables for better readability and maintainability
set(CUDA_ROOT /usr/local/cuda)  # Path to CUDA installation (adjust as needed)
set(CMAKE_CUDA_STANDARD 17)     # Specify the CUDA standard to use
set(CMAKE_CUDA_STANDARD_REQUIRED ON)  # Ensure the CUDA standard is required
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80)  # Target GPU architectures

# Enable detailed FetchContent logging for clarity
set(FETCHCONTENT_QUIET OFF)

# Include FetchContent module for external dependency management
include(FetchContent)

# Fetch NVIDIA CUDA samples repository for common utilities
FetchContent_Declare(
        CudaDependencies
        GIT_REPOSITORY https://github.com/NVIDIA/cuda-samples.git
        GIT_TAG master
)

# Populate the fetched content (downloads and prepares the dependency)
FetchContent_MakeAvailable(CudaDependencies)

# Include the common utilities directory from the fetched CUDA samples
include_directories(${FETCHCONTENT_BASE_DIR}/cudadependencies-src/Common)

# Include the CUDA include directory for required headers
include_directories(${CUDA_ROOT}/include)

# Add the source file for the Gauss filter as the main executable target
add_executable(${PROJECT_NAME} gauss_filter.cu)

# Link required CUDA libraries
target_link_libraries(${PROJECT_NAME}
        freeimage                                # Image processing library
        ${CUDA_ROOT}/lib64/libcudart.so          # CUDA runtime
        ${CUDA_ROOT}/lib64/libnppc.so            # Core NPP library
        ${CUDA_ROOT}/lib64/libnppial.so          # NPP linear algebra
        ${CUDA_ROOT}/lib64/libnppif.so           # NPP filtering
        ${CUDA_ROOT}/lib64/libnppicc.so          # NPP color conversion
        ${CUDA_ROOT}/lib64/libnppig.so           # NPP geometry
        ${CUDA_ROOT}/lib64/libnppisu.so          # NPP signal processing
)

# Suppress warnings about deprecated GPU targets
add_compile_options(-Wno-deprecated-gpu-targets)

# Print a summary of key configurations for transparency
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "CUDA Root Directory: ${CUDA_ROOT}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Using FetchContent for CUDA Dependencies from NVIDIA CUDA Samples.")