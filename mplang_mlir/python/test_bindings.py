#!/usr/bin/env python3
"""Test script for Mpir Python bindings.

This script tests basic functionality of the Mpir dialect Python bindings.
Run after building with MLIR_ENABLE_BINDINGS_PYTHON=ON.
"""

import sys


def test_import():
    """Test that we can import the mplang_mlir module."""
    try:
        import mplang_mlir  # noqa: F401

        print("✓ Successfully imported mplang_mlir")
        return True
    except ImportError as e:
        print(f"✗ Failed to import mplang_mlir: {e}")
        return False


def test_dialect_registration():
    """Test dialect registration with MLIR context."""
    try:
        from mplang_mlir.dialects import mpir

        from mplang_mlir import ir

        ctx = ir.Context()
        mpir.register_dialect(ctx)
        print("✓ Successfully registered Mpir dialect")
        return True
    except Exception as e:
        print(f"✗ Failed to register dialect: {e}")
        return False


def test_dialect_load():
    """Test loading dialect into context."""
    try:
        from mplang_mlir.dialects import mpir

        from mplang_mlir import ir

        ctx = ir.Context()
        mpir.load_dialect(ctx)
        print("✓ Successfully loaded Mpir dialect")
        return True
    except Exception as e:
        print(f"✗ Failed to load dialect: {e}")
        return False


def test_parse_peval():
    """Test parsing a peval operation."""
    try:
        from mplang_mlir.dialects import mpir

        from mplang_mlir import ir

        ctx = ir.Context()
        mpir.register_dialect(ctx)
        ctx.allow_unregistered_dialects = True

        module_str = """
        module {
          func.func @simple() {
            %mask = arith.constant dense<true> : tensor<3xi1>
            %arg0 = arith.constant dense<1.0> : tensor<3xf32>
            %result = mpir.peval @callee(%arg0, %mask) : (tensor<3xf32>, tensor<3xi1>) -> tensor<3xf32>
            return
          }
        }
        """

        module = ir.Module.parse(module_str, ctx)
        print("✓ Successfully parsed peval operation")
        print(f"  Module: {module}")
        return True
    except Exception as e:
        print(f"✗ Failed to parse peval: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Mpir Python bindings")
    print("=" * 60)

    tests = [
        test_import,
        test_dialect_registration,
        test_dialect_load,
        test_parse_peval,
    ]

    results = [test() for test in tests]

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
