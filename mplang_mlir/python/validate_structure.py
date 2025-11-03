#!/usr/bin/env python3
"""Validate the extended Python bindings structure."""

import ast
import sys
from pathlib import Path


def check_file_syntax(filepath):
    """Check if Python file has valid syntax."""
    print(f"\nChecking {filepath.name}...")
    try:
        with open(filepath) as f:
            code = f.read()
        ast.parse(code)
        print("  ✓ Syntax valid")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def check_extended_operations(filepath):
    """Check that all extended operations are defined."""
    print(f"\nAnalyzing extended operations in {filepath.name}...")

    with open(filepath) as f:
        code = f.read()

    expected_ops = [
        "PEvalOp",
        "PEvalDynOp",
        "UniformCondOp",
        "UniformWhileOp",
        "ConvOp",
        "ShflSOp",
    ]

    found_ops = []
    for op in expected_ops:
        pattern = f"class {op}("
        if pattern in code:
            found_ops.append(op)
            print(f"  ✓ {op} defined")
        else:
            print(f"  ✗ {op} missing")

    return len(found_ops) == len(expected_ops)


def check_register_decorators(filepath):
    """Check that operations use @register_operation decorator."""
    print(f"\nChecking decorators in {filepath.name}...")

    with open(filepath) as f:
        lines = f.readlines()

    decorator_count = 0
    for i, line in enumerate(lines):
        if "@_cext.register_operation" in line:
            decorator_count += 1
            # Check next line for class definition
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if "class" in next_line:
                    class_name = next_line.split("class")[1].split("(")[0].strip()
                    print(f"  ✓ Decorator for {class_name}")

    print(f"\n  Total decorators: {decorator_count}")
    return decorator_count >= 5  # At least 5 operations


def check_docstrings(filepath):
    """Check that operations have docstrings."""
    print(f"\nChecking docstrings in {filepath.name}...")

    tree = ast.parse(open(filepath).read())

    docstring_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.endswith("Op"):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstring_count += 1
                    print(f"  ✓ {node.name} has docstring")
                else:
                    print(f"  ✗ {node.name} missing docstring")

    return docstring_count >= 5


def check_type_hints(filepath):
    """Check that __init__ methods have type hints."""
    print(f"\nChecking type hints in {filepath.name}...")

    with open(filepath) as f:
        code = f.read()

    # Simple check for type annotations
    has_sequence_type = "Sequence[Type]" in code
    has_sequence_value = "Sequence[Value]" in code
    has_value_type = ": Value" in code

    if has_sequence_type:
        print("  ✓ Uses Sequence[Type] annotation")
    if has_sequence_value:
        print("  ✓ Uses Sequence[Value] annotation")
    if has_value_type:
        print("  ✓ Uses Value type annotation")

    return has_sequence_type and has_sequence_value


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("Extended Python Bindings - Structure Validation")
    print("=" * 70)

    # Find the extension file
    ext_file = Path(__file__).parent / "mplang_mlir" / "dialects" / "_mpir_ops_ext.py"

    if not ext_file.exists():
        print(f"\n✗ Extension file not found: {ext_file}")
        return False

    print(f"\nFound extension file: {ext_file}")

    # Run checks
    checks = {
        "Syntax": check_file_syntax(ext_file),
        "Operations": check_extended_operations(ext_file),
        "Decorators": check_register_decorators(ext_file),
        "Docstrings": check_docstrings(ext_file),
        "Type Hints": check_type_hints(ext_file),
    }

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)

    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(checks.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All checks passed!")
        print("\nNext steps:")
        print("  1. Build MLIR with Python bindings (MLIR_ENABLE_BINDINGS_PYTHON=ON)")
        print("  2. Run actual tests: python test_ops_ext.py")
        print("  3. Implement AST-to-MLIR converter")
    else:
        print("✗ Some checks failed")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
