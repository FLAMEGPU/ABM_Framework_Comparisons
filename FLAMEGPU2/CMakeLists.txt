# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)

# Optionaly set the version of flamegpu which should be used.
# Use a branch name such as `master`, or a tagged version such as `v2.0.0-alpha`
set(FLAMEGPU_VERSION "v2.0.0-rc" CACHE STRING "FLAMEGPU/FLAMEGPU2 git branch or tag to use")
# If the above version is a hash instead, also set FLAMEGPU_VERSION_ALLOW_HASH to ON
# set(FLAMEGPU_VERSION_ALLOW_HASH "ON")

# Find the FLAMEGPU 2 and import it's CMake rules
include(${CMAKE_CURRENT_LIST_DIR}/cmake/flamegpu2.cmake)

# Handle CMAKE_CUDA_ARCHITECTURES gracefully, passign the project name for code-injection
include(${FLAMEGPU_ROOT}/cmake/CUDAArchitectures.cmake)
flamegpu_init_cuda_architectures(PROJECT schelling)

# Define output location of binary files
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)

# ---------
# Define the build rules for the Schelling implemetnation (@todo - sub CMakeLists.txt?)

# Name the project and set languages
project(schelling CUDA CXX)

include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Manually list relevant header and source files
SET(ALL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/schelling.cu
)

# Declare teh executable using a FLAMEGPU provided CMake function
flamegpu_add_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" TRUE)

# ------
# Define the build rules for the Boids 2D implementation

# Handle CMAKE_CUDA_ARCHITECTURES gracefully, passign the project name for code-injection
flamegpu_init_cuda_architectures(PROJECT boids2D)
# Name the project and set languages
project(boids2D CUDA CXX)

include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Manually list relevant header and source files
SET(ALL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/boids2D.cu
)

# Declare teh executable using a FLAMEGPU provided CMake function
flamegpu_add_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" TRUE)