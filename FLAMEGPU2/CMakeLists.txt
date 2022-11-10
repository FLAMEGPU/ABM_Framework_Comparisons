# Set the minimum cmake version requires by this project.
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

# Option to enable/disable building the visualisation
option(VISUALISATION "Enable visualisation support" OFF)

# Optionaly set the version of flamegpu which should be used.
# Use a branch name such as `master`, or a tagged version such as `v2.0.0-alpha`
# @todo - DO NOT MERGE THIS VERSION, THIS HASH WILL DISSAPEAR
set(FLAMEGPU_VERSION "10213d5dac58cf238fae0b28a51f10b8085df106" CACHE STRING "FLAMEGPU/FLAMEGPU2 git branch or tag to use")
# If the above version is a hash instead, also set FLAMEGPU_VERSION_ALLOW_HASH to ON
set(FLAMEGPU_VERSION_ALLOW_HASH "ON")

# Find the FLAMEGPU 2 and import it's CMake rules
include(${CMAKE_CURRENT_LIST_DIR}/cmake/flamegpu2.cmake)

# ---------
# Define the build rules for the Schelling implemetnation (@todo - sub CMakeLists.txt?)

# Name the project and set languages
project(schelling CUDA CXX)

# Define output location of binary files
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
else()
    # If called via add_subdirectory()
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/${CMAKE_BUILD_TYPE}/)
endif()

# Manually list relevant header and source files
SET(ALL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/schelling.cu
)

# Declare teh executable using a FLAMEGPU provided CMake function
add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" TRUE)

# ------
# Define the build rules for the Boids 2D implementation


# Name the project and set languages
project(boids2D CUDA CXX)

# Define output location of binary files
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
else()
    # If called via add_subdirectory()
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/${CMAKE_BUILD_TYPE}/)
endif()

include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Manually list relevant header and source files
SET(ALL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/boids2D.cu
)

# Declare teh executable using a FLAMEGPU provided CMake function
add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" TRUE)