cmake_minimum_required(VERSION 3.20)
project(MPLANG_Dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expect MLIR to be pre-installed; pass -DMLIR_DIR=/path/to/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig at: ${MLIR_DIR}")

# MLIR packages expose the LLVM CMake package directory as LLVM_CMAKE_DIR.
# We need both LLVM and MLIR CMake modules for add_mlir_library, tablegen, etc.
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVMConfig at: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(HandleLLVMOptions)
include(TableGen)
include(AddMLIR)

add_subdirectory(lib/Dialect/Mplang)
add_subdirectory(tools/mplang-opt)
