#!/bin/bash
# Test runner for Mpir dialect tests
# Usage: ./run_tests.sh [test_name]
#   If test_name is provided, only run that specific test
#   Otherwise, run all tests in test/Dialect/Mpir/

set -e

# Determine script directory (mplang_mlir root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/built"
TEST_DIR="$SCRIPT_DIR/test/Dialect/Mpir"
MPIR_OPT="$BUILD_DIR/tools/mpir-opt/mpir-opt"

# Check if mpir-opt exists
if [ ! -f "$MPIR_OPT" ]; then
    echo "Error: mpir-opt not found at $MPIR_OPT"
    echo "Please build the project first: cd $BUILD_DIR && ninja mpir-opt"
    exit 1
fi

echo "Running Mpir dialect tests..."
echo "================================"
echo

TOTAL=0
PASSED=0
FAILED=0

# Determine which tests to run
if [ -n "$1" ]; then
    # Run specific test
    TEST_FILES=("$TEST_DIR/$1")
    if [ ! -f "${TEST_FILES[0]}" ]; then
        echo "Error: Test file not found: $1"
        echo "Available tests:"
        ls -1 "$TEST_DIR"/*.mlir 2>/dev/null | xargs -n1 basename || echo "  (none)"
        exit 1
    fi
else
    # Run all tests
    TEST_FILES=("$TEST_DIR"/*.mlir)
fi

# Run each test file
for test_file in "${TEST_FILES[@]}"; do
    test_name=$(basename "$test_file")
    TOTAL=$((TOTAL + 1))

    echo "Testing: $test_name"

    if "$MPIR_OPT" "$test_file" -split-input-file -verify-diagnostics > /dev/null 2>&1; then
        echo "  ✓ PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  ✗ FAILED"
        FAILED=$((FAILED + 1))
        echo "  Re-running with output:"
        "$MPIR_OPT" "$test_file" -split-input-file -verify-diagnostics || true
    fi
    echo
done

echo "================================"
echo "Test Summary:"
echo "  Total:  $TOTAL"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo

if [ $FAILED -eq 0 ]; then
    echo "All tests passed! ✓"
    exit 0
else
    echo "Some tests failed! ✗"
    exit 1
fi
