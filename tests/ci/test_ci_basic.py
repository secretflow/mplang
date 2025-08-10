#!/usr/bin/env python3
"""
Basic CI smoke test that doesn't require complex dependencies.
This test validates basic Python functionality and project structure.
"""

import os
import sys
from pathlib import Path

# Import pytest if available for proper test discovery
try:
    import pytest
except ImportError:
    pytest = None


def test_project_structure():
    """Test that the basic project structure exists."""
    # Go up two levels from tests/ci/ to get to project root
    project_root = Path(__file__).parent.parent.parent
    
    # Check essential directories exist
    assert (project_root / "mplang").exists(), "mplang directory missing"
    assert (project_root / "tests").exists(), "tests directory missing"
    assert (project_root / "examples").exists(), "examples directory missing"
    assert (project_root / "tutorials").exists(), "tutorials directory missing"
    
    # Check essential files exist
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml missing"
    assert (project_root / "README.md").exists(), "README.md missing"
    assert (project_root / "mplang" / "__init__.py").exists(), "mplang/__init__.py missing"


def test_python_version():
    """Test that we're running on a supported Python version."""
    version = sys.version_info
    assert version >= (3, 10), f"Python 3.10+ required, got {version.major}.{version.minor}"


def test_can_import_basic_modules():
    """Test importing basic Python modules used by the project."""
    try:
        import pathlib
        import collections.abc
        import dataclasses
        import functools
        print("✓ Basic Python modules imported successfully")
    except ImportError as e:
        if pytest:
            pytest.fail(f"Failed to import basic Python modules: {e}")
        else:
            raise AssertionError(f"Failed to import basic Python modules: {e}")


def test_pyproject_toml_valid():
    """Test that pyproject.toml is valid."""
    try:
        import tomllib
    except ImportError:
        # Python < 3.11 fallback
        try:
            import tomli as tomllib
        except ImportError:
            print("Warning: Cannot validate pyproject.toml - tomllib/tomli not available")
            return
    
    # Go up two levels from tests/ci/ to get to project root
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    # Basic validation
    assert "project" in data, "Missing [project] section in pyproject.toml"
    assert "name" in data["project"], "Missing project name"
    assert data["project"]["name"] == "mplang", "Incorrect project name"
    assert "version" in data["project"], "Missing project version"


if __name__ == "__main__":
    # Run tests manually if called directly
    test_project_structure()
    test_python_version()
    test_can_import_basic_modules()
    test_pyproject_toml_valid()
    print("✓ All basic CI tests passed!")