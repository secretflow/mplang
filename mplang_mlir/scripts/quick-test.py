#!/usr/bin/env python3
"""Quick test to verify MPLang + MLIR development environment."""

import sys


def test_imports():
    """Test basic imports."""
    print("🧪 Testing imports...")
    results = []

    # Test mplang
    try:
        import mplang  # noqa: F401

        print("  ✓ mplang")
        results.append(("mplang", True, None))
    except ImportError as e:
        print(f"  ✗ mplang: {e}")
        results.append(("mplang", False, e))

    # Test mplang_mlir (optional)
    try:
        from mplang_mlir.dialects import mpir  # noqa: F401

        import mplang_mlir  # noqa: F401

        print("  ✓ mplang_mlir")
        print("  ✓ mplang_mlir.dialects.mpir")
        results.append(("mplang_mlir", True, None))
    except ImportError as e:
        print(f"  ⚠ mplang_mlir: {e}")
        print("    (Build C++ dialect if you need MLIR backend)")
        results.append(("mplang_mlir", False, e))

    return results


def test_mplang_features():
    """Test basic mplang features."""
    print("\n🔧 Testing MPLang features...")

    try:
        from mplang import compile, function

        @function
        def simple_add(x, y):
            return x + y

        print("  ✓ @function decorator works")

        # Try compilation (without execution)
        try:
            _result = compile(simple_add)
            print("  ✓ compile() works")
        except Exception as e:
            print(f"  ⚠ compile() failed: {e}")

        return True
    except Exception as e:
        print(f"  ✗ MPLang features: {e}")
        return False


def test_mlir_integration():
    """Test MLIR integration (if available)."""
    print("\n🎯 Testing MLIR integration...")

    try:
        from mlir import ir
        from mplang_mlir.dialects import mpir

        import mplang_mlir  # noqa: F401

        # Create a simple MLIR context
        with ir.Context() as ctx:
            mpir.register_dialect(ctx)
            loc = ir.Location.unknown()
            _module = ir.Module.create(loc)
            print("  ✓ MLIR context creation works")
            print("  ✓ Mpir dialect registered")

        return True
    except ImportError:
        print("  ⚠ MLIR bindings not available")
        return False
    except Exception as e:
        print(f"  ✗ MLIR integration error: {e}")
        return False


def print_environment():
    """Print environment information."""
    print("\n📊 Environment Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    print("\n  PYTHONPATH:")
    for i, path in enumerate(sys.path[:5]):
        if i == 0:
            print(f"    {path} (cwd)")
        else:
            print(f"    {path}")
    if len(sys.path) > 5:
        print(f"    ... and {len(sys.path) - 5} more")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MPLang + MLIR Development Environment Test")
    print("=" * 60)

    # Test imports
    import_results = test_imports()

    # Test MPLang features
    mplang_ok = False
    if any(pkg == "mplang" and success for pkg, success, _ in import_results):
        mplang_ok = test_mplang_features()

    # Test MLIR integration
    mlir_ok = False
    if any(pkg == "mplang_mlir" and success for pkg, success, _ in import_results):
        mlir_ok = test_mlir_integration()

    # Print environment
    print_environment()

    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary:")
    for pkg, success, _error in import_results:
        status = "✓" if success else ("⚠" if pkg == "mplang_mlir" else "✗")
        print(f"  {status} {pkg}: {'OK' if success else 'Not available'}")

    if mplang_ok:
        print("  ✓ MPLang features: OK")

    if mlir_ok:
        print("  ✓ MLIR integration: OK")

    print("=" * 60)

    # Exit code
    core_ok = any(pkg == "mplang" and success for pkg, success, _ in import_results)
    if core_ok:
        print("\n✅ Development environment is ready!")
        if not mlir_ok:
            print("   (MLIR backend optional - build when needed)")
        return 0
    else:
        print("\n❌ Core mplang not available!")
        print("   Make sure PYTHONPATH includes repo root")
        return 1


if __name__ == "__main__":
    sys.exit(main())
