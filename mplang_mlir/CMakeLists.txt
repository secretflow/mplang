cmake_minimum_required(VERSION 3.20)
project(MPLANG LANGUAGES C CXX)

if(NOT DEFINED MLIR_DIR)
	message(FATAL_ERROR "MLIR_DIR is required. Run scripts/mlir/setup_mlir.sh first or set -DMLIR_DIR explicitly.")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found LLVM: ${LLVM_DIR}")
message(STATUS "Found MLIR: ${MLIR_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR} ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Enable CTest for add_test() below
include(CTest)
enable_testing()

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
