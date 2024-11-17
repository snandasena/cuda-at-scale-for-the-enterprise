
# Gauss Filter with CUDA and NPP

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
    - [Linux](#linux)
    - [Windows](#windows)
5. [Usage](#usage)
    - [Command-line Arguments](#command-line-arguments)
    - [Default Directories](#default-directories)
    - [Example](#example)
6. [Clean Up](#clean-up)
7. [CMake Configuration](#cmake-configuration)
8. [Contributions](#contributions)

## Overview
This project demonstrates image processing using CUDA and NVIDIA Performance Primitives (NPP) to apply a Gaussian filter on images. It showcases parallel processing by applying the filter to multiple images concurrently using threads and saves the processed images in the output directory.

## Features
- Applies Gaussian filter on input images using NPP.
- Supports batch processing with multithreading for faster performance.
- Configurable input and output directories via command-line arguments.

## Requirements
- **CUDA Toolkit** (version 11.4 or later)
- **FreeImage** library for image I/O operations
- C++17 compatible compiler
- NVIDIA GPU with Compute Capability 3.5 or higher

## Installation

### Linux
1. Install the required libraries:
    ```bash
    sudo apt-get install libfreeimage-dev
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/snandasena/cuda-at-scale-for-the-enterprise.git
    cd cuda-at-scale-for-the-enterprise
    ```

3. Build the project:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

### Windows
- I haven't tried on Windows platforms due to dependencies issues.
- TODO: Try on Windows later

## Usage

### Command-line Arguments
You can specify custom input and output directories:
```bash
./GaussFilter --input /path/to/input --output /path/to/output
```

- **`--input`**: Path to the input directory containing `.bmp` images.
- **`--output`**: Path to the output directory for the filtered images.

### Default Directories
If no arguments are provided, the default directories are:
- **Input Directory**: `data/`
- **Output Directory**: `output/`

### Example:
```bash
./GaussFilter --input ../data/ --output ../output/
```

## Clean Up
To clean up the build directory:
```bash
make clean
```

## CMake Configuration

- **CUDA**: Supports multiple GPU architectures (compute capabilities 5.0 to 8.0).
- **External Dependencies**: Uses FreeImage for image processing and NPP for GPU acceleration.

### CMake Options:
- `CMAKE_CUDA_ARCHITECTURES`: Specifies the supported CUDA architectures.
- `FETCHCONTENT_QUIET`: Enables detailed logging for external dependencies.

## Contributions
Feel free to fork, contribute, or file issues for bug fixes and feature improvements.

---

### Code Summary:

#### **File: `gauss_filter.cu`**
- **Libraries Used**:
    - CUDA Runtime
    - NPP (NVIDIA Performance Primitives)
    - FreeImage
- **Main Components**:
    - **Gaussian Filter Application**: Uses NPP to filter `.bmp` images with a Gaussian kernel.
    - **Multithreading**: Processes images in parallel for faster execution.
    - **Directory Management**: Input and output directories are customizable via command-line flags.
    - **Error Handling**: Catches and reports exceptions during image processing.

#### **Functions:**
- **`printfNPPinfo()`**: Prints detailed information about CUDA and NPP versions.
- **`applyGaussFilter()`**: Applies a Gaussian filter to a single image.
- **`cleanupOutputDirectory()`**: Deletes all files in the output directory before processing.
- **`processBatch()`**: Processes a batch of images concurrently using threads.
- **`processImagesInDirectory()`**: Processes all images in a given directory, organizing them into batches for parallel processing.
- **`parseInputOutputDirs()`**: Parses command-line arguments to get custom input and output directories.

#### **CMake Configuration:**
The project is configured with CMake for CUDA, ensuring compatibility with multiple GPU architectures. It also integrates FreeImage for image loading and saving.

```cmake
cmake_minimum_required(VERSION 3.30)

# Set CMake policy CMP0104 to NEW to ensure CUDA architectures are set properly
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
```

