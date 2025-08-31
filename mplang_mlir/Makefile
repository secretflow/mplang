# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Makefile for mplang_mlir

# Variables
ROOT_DIR := $(shell git rev-parse --show-toplevel)
BUILD_DIR := $(ROOT_DIR)/build
MPLANG_MLIR_BUILD_DIR := $(BUILD_DIR)/mplang_mlir

# Default MLIR/LLVM directories, can be overridden from environment
MLIR_DIR ?= $(ROOT_DIR)/build/llvm/lib/cmake/mlir
LLVM_DIR ?= $(ROOT_DIR)/build/llvm/lib/cmake/llvm

# Tools
CMAKE := cmake
CTEST := ctest

# Main binary
MPLANG_OPT := $(MPLANG_MLIR_BUILD_DIR)/tools/mplang-opt/mplang-opt

# Phony targets for commands
.PHONY: all configure build test run clean help

# Default target
all: build

help:
	@echo "Makefile for mplang_mlir"
	@echo ""
	@echo "Usage:"
	@echo "  make <target> [VAR=VALUE]"
	@echo ""
	@echo "Targets:"
	@echo "  all         Build the project (default)."
	@echo "  configure   Configure the project using CMake."
	@echo "  build       Build the mplang-opt binary."
	@echo "  test        Run tests using ctest."
	@echo "  run         Run a sample peval test with mplang-opt."
	@echo "  clean       Remove the build directory."
	@echo "  help        Show this help message."
	@echo ""
	@echo "Variables:"
	@echo "  MLIR_DIR    Path to MLIR CMake directory. Default: $(MLIR_DIR)"
	@echo "  LLVM_DIR    Path to LLVM CMake directory. Default: $(LLVM_DIR)"

# Configure the project using CMake
# Creates the build directory and generates build files.
configure:
	@mkdir -p $(MPLANG_MLIR_BUILD_DIR)
	@echo "--- Configuring project in $(MPLANG_MLIR_BUILD_DIR) ---"
	$(CMAKE) -S . -B $(MPLANG_MLIR_BUILD_DIR) -G Ninja \
		-DMLIR_DIR=$(MLIR_DIR) \
		-DLLVM_DIR=$(LLVM_DIR)

# Build the project
# This target depends on the build.ninja file, which is created by 'configure'.
# If it doesn't exist, 'make' will automatically run the 'configure' target first.
build: $(MPLANG_MLIR_BUILD_DIR)/build.ninja
	@echo "--- Building mplang-opt ---"
	$(CMAKE) --build $(MPLANG_MLIR_BUILD_DIR) --target mplang-opt -j

$(MPLANG_MLIR_BUILD_DIR)/build.ninja:
	$(MAKE) configure

# Run tests
test: build
	@echo "--- Running tests ---"
	$(CTEST) --test-dir $(MPLANG_MLIR_BUILD_DIR) --output-on-failure

# Run a sample command
run: build
	@echo "--- Running peval example ---"
	$(MPLANG_OPT) --mpir-peval test/mlir/mpir/peval.mlir

# Clean the build directory
clean:
	@echo "--- Cleaning build directory: $(MPLANG_MLIR_BUILD_DIR) ---"
	@rm -rf $(MPLANG_MLIR_BUILD_DIR)
