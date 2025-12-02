# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for func_utils module type definitions.
"""

import numpy as np
import pytest
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.v1.utils.func_utils import (
    MorphStruct,
    is_treedef_list,
    validate_morph_struct,
    var_demorph,
    var_morph,
)


def test_morph_struct_type():
    """Test MorphStruct type alias."""
    # Create some test data
    data = {"a": [1, 2], "b": 3}
    _flat_data, tree_def = tree_flatten(data)

    # Create a MorphStruct
    morph_struct: MorphStruct = (tree_def, (0, 1))

    # Verify it's a tuple with correct types
    assert isinstance(morph_struct, tuple)
    assert len(morph_struct) == 2

    # The first element should be a PyTreeDef
    assert isinstance(morph_struct[0], PyTreeDef)

    # The second element should be a tuple of ints
    assert isinstance(morph_struct[1], tuple)
    assert all(isinstance(x, int) for x in morph_struct[1])


def test_validate_morph_struct():
    """Test validate_morph_struct function."""
    # Test valid MorphStruct
    data = {"a": [1, 2], "b": 3}
    _flat_data, tree_def = tree_flatten(data)
    valid_morph_struct: MorphStruct = (tree_def, (0, 1))

    # Should not raise any exception
    validate_morph_struct(valid_morph_struct)

    # Test invalid cases
    with pytest.raises(TypeError, match="MorphStruct must be a tuple of"):
        validate_morph_struct("not a tuple")  # type: ignore

    with pytest.raises(TypeError, match="MorphStruct must be a tuple of"):
        validate_morph_struct((tree_def,))  # type: ignore

    with pytest.raises(TypeError, match="MorphStruct\\[0\\] must be PyTreeDef"):
        validate_morph_struct(("not a PyTreeDef", (0, 1)))  # type: ignore

    with pytest.raises(TypeError, match="MorphStruct\\[1\\] must be tuple"):
        validate_morph_struct((tree_def, [0, 1]))  # type: ignore

    with pytest.raises(
        TypeError, match="MorphStruct\\[1\\] must contain only integers"
    ):
        validate_morph_struct((tree_def, (0, "not an int")))  # type: ignore


def create_empty_morph_struct() -> MorphStruct:
    """Create an empty MorphStruct for testing purposes.

    Returns a MorphStruct representing an empty structure with no variables.
    This is useful for test cases where you need a valid MorphStruct but
    don't actually have any complex structure to represent.
    """
    # Create a simple empty tuple and flatten it
    _, tree_def = tree_flatten(())
    return (tree_def, ())


def test_create_empty_morph_struct():
    """Test create_empty_morph_struct function."""
    empty_struct = create_empty_morph_struct()

    # Should be a valid MorphStruct
    validate_morph_struct(empty_struct)

    # Should represent empty structure
    tree_def, split_info = empty_struct
    assert isinstance(tree_def, PyTreeDef)
    assert split_info == ()


def test_var_morph_returns_morphstruct():
    """Test that var_morph returns proper MorphStruct."""
    # Test with mixed data
    data = ([1, 2], {"c": 3})
    is_variable = lambda x: isinstance(x, list)

    variables, immediates, morph_struct = var_morph(data, is_variable)

    # Verify types
    assert isinstance(variables, list)
    assert isinstance(immediates, list)
    assert isinstance(morph_struct, tuple)
    assert len(morph_struct) == 2

    # Verify the MorphStruct is correctly typed
    tree_def, split_info = morph_struct
    assert isinstance(tree_def, PyTreeDef)
    assert isinstance(split_info, tuple)
    assert all(isinstance(x, int) for x in split_info)

    # Test that it passes validation
    validate_morph_struct(morph_struct)


def test_var_demorph_with_morphstruct():
    """Test that var_demorph works correctly with MorphStruct."""
    # Create test data
    original_data = ([np.array([1, 2])], {"scalar": 42})
    is_variable = lambda x: isinstance(x, np.ndarray)

    # Morph the data
    variables, immediates, morph_struct = var_morph(original_data, is_variable)

    # Verify we can demorph it back
    reconstructed = var_demorph(variables, immediates, morph_struct)

    # Check structure is preserved
    assert len(reconstructed) == 2  # tuple with 2 elements
    assert isinstance(reconstructed[0], list)
    assert isinstance(reconstructed[1], dict)

    # Check content
    assert np.array_equal(reconstructed[0][0], original_data[0][0])
    assert reconstructed[1]["scalar"] == original_data[1]["scalar"]


def test_empty_var_morph():
    """Test var_morph with no variables."""
    data = (1, 2, "hello")
    is_variable = lambda x: False  # Nothing is a variable

    variables, immediates, morph_struct = var_morph(data, is_variable)

    assert variables == []
    assert immediates == [1, 2, "hello"]

    # Verify MorphStruct is still well-formed
    tree_def, split_info = morph_struct
    assert isinstance(tree_def, PyTreeDef)
    assert isinstance(split_info, tuple)
    assert split_info == ()  # Empty tuple for no variables

    # Test that it passes validation
    validate_morph_struct(morph_struct)


def test_all_variables_morph():
    """Test var_morph with all variables."""
    data = [np.array([1]), np.array([2]), np.array([3])]
    is_variable = lambda x: isinstance(x, np.ndarray)

    variables, immediates, morph_struct = var_morph(data, is_variable)

    assert len(variables) == 3
    assert immediates == []

    # Verify MorphStruct
    tree_def, split_info = morph_struct
    assert isinstance(tree_def, PyTreeDef)
    assert isinstance(split_info, tuple)
    assert split_info == (0, 1, 2)  # All positions are variables

    # Test that it passes validation
    validate_morph_struct(morph_struct)


def test_is_treedef_list_simple_lists():
    """Test that simple lists are recognized as TreeDef lists."""
    # Simple list of scalars
    simple_list = [1, 2, 3]
    _, tree_def = tree_flatten(simple_list)
    assert is_treedef_list(tree_def), "Simple list should be recognized as TreeDef list"

    # Simple list of arrays
    array_list = [np.array([1, 2]), np.array([3, 4])]
    _, tree_def = tree_flatten(array_list)
    assert is_treedef_list(tree_def), (
        "List of arrays should be recognized as TreeDef list"
    )

    # Empty list
    empty_list = []
    _, tree_def = tree_flatten(empty_list)
    assert is_treedef_list(tree_def), "Empty list should be recognized as TreeDef list"

    # Single element list
    single_list = [42]
    _, tree_def = tree_flatten(single_list)
    assert is_treedef_list(tree_def), (
        "Single element list should be recognized as TreeDef list"
    )


def test_is_treedef_list_nested_structures():
    """Test that nested structures are not recognized as TreeDef lists."""
    # Nested list
    nested_list = [[1, 2], [3, 4]]
    _, tree_def = tree_flatten(nested_list)
    assert not is_treedef_list(tree_def), (
        "Nested list should not be recognized as TreeDef list"
    )

    # Dictionary
    dict_structure = {"a": 1, "b": 2}
    _, tree_def = tree_flatten(dict_structure)
    assert not is_treedef_list(tree_def), (
        "Dictionary should not be recognized as TreeDef list"
    )

    # Tuple
    tuple_structure = (1, 2, 3)
    _, tree_def = tree_flatten(tuple_structure)
    assert not is_treedef_list(tree_def), (
        "Tuple should not be recognized as TreeDef list"
    )

    # Complex nested structure
    complex_structure = {"data": [1, 2], "config": {"param": 3}}
    _, tree_def = tree_flatten(complex_structure)
    assert not is_treedef_list(tree_def), (
        "Complex nested structure should not be recognized as TreeDef list"
    )


def test_is_treedef_list_args_kwargs_structure():
    """Test that args/kwargs structure is not recognized as TreeDef list."""
    # (args, kwargs) tuple structure
    args_kwargs = ([1, 2, 3], {"param": 4})
    _, tree_def = tree_flatten(args_kwargs)
    assert not is_treedef_list(tree_def), (
        "Args/kwargs structure should not be recognized as TreeDef list"
    )

    # Empty args with kwargs
    empty_args_kwargs = ([], {"param": 4})
    _, tree_def = tree_flatten(empty_args_kwargs)
    assert not is_treedef_list(tree_def), (
        "Empty args with kwargs should not be recognized as TreeDef list"
    )

    # Args with empty kwargs
    args_empty_kwargs = ([1, 2], {})
    _, tree_def = tree_flatten(args_empty_kwargs)
    assert not is_treedef_list(tree_def), (
        "Args with empty kwargs should not be recognized as TreeDef list"
    )


def test_is_treedef_list_non_list_containers():
    """Test that non-list containers are not recognized as TreeDef lists."""
    # Named tuple
    from typing import NamedTuple

    class Point(NamedTuple):
        x: int
        y: int

    point = Point(1, 2)
    _, tree_def = tree_flatten(point)
    assert not is_treedef_list(tree_def), (
        "Named tuple should not be recognized as TreeDef list"
    )


def test_is_treedef_list_mixed_types():
    """Test that lists with mixed types are still recognized as TreeDef lists."""
    # List with mixed types (all leaf nodes)
    mixed_list = [1, "hello", 3.14, True]
    _, tree_def = tree_flatten(mixed_list)
    assert is_treedef_list(tree_def), (
        "List with mixed leaf types should be recognized as TreeDef list"
    )

    # List with arrays and scalars
    mixed_array_list = [np.array([1, 2]), 42, "test"]
    _, tree_def = tree_flatten(mixed_array_list)
    assert is_treedef_list(tree_def), (
        "List with mixed arrays and scalars should be recognized as TreeDef list"
    )


def test_is_treedef_list_complex_elements():
    """Test that lists containing elements with multiple leaves are not TreeDef lists."""
    # List containing structures with multiple values each
    list_with_multi_value_dicts = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
    _, tree_def = tree_flatten(list_with_multi_value_dicts)
    assert not is_treedef_list(tree_def), (
        "List with multi-value dict elements should not be recognized as TreeDef list"
    )

    # List containing lists (each sublist has multiple leaves)
    list_with_lists = [[1, 2], [3, 4]]
    _, tree_def = tree_flatten(list_with_lists)
    assert not is_treedef_list(tree_def), (
        "List with list elements should not be recognized as TreeDef list"
    )

    # List containing tuples (each tuple has multiple leaves)
    list_with_tuples = [(1, 2), (3, 4)]
    _, tree_def = tree_flatten(list_with_tuples)
    assert not is_treedef_list(tree_def), (
        "List with tuple elements should not be recognized as TreeDef list"
    )


def test_is_treedef_list_single_value_elements():
    """Test that lists where each element contributes exactly one leaf are TreeDef lists."""
    # List containing single-key dictionaries (each dict has 1 leaf)
    list_with_single_dicts = [{"value": 1}, {"value": 2}, {"value": 3}]
    _, tree_def = tree_flatten(list_with_single_dicts)
    assert is_treedef_list(tree_def), (
        "List with single-value dict elements should be recognized as TreeDef list"
    )

    # List containing single-element tuples (each tuple has 1 leaf)
    list_with_single_tuples = [(1,), (2,), (3,)]
    _, tree_def = tree_flatten(list_with_single_tuples)
    assert is_treedef_list(tree_def), (
        "List with single-element tuple elements should be recognized as TreeDef list"
    )


def test_is_treedef_list_edge_cases():
    """Test edge cases for is_treedef_list function."""
    # List with None values (JAX treats None as having 0 leaves, so not a TreeDef list)
    list_with_none = [1, None, 3]
    _, tree_def = tree_flatten(list_with_none)
    assert not is_treedef_list(tree_def), (
        "List with None values should not be recognized as TreeDef list (None has 0 leaves)"
    )

    # Empty list (no children, so all() returns True)
    empty_list = []
    _, tree_def = tree_flatten(empty_list)
    assert is_treedef_list(tree_def), "Empty list should be recognized as TreeDef list"

    # Deeply nested structure that's clearly not a TreeDef list
    deep_nested = {"level1": {"level2": {"level3": [1]}}}
    _, tree_def = tree_flatten(deep_nested)
    assert not is_treedef_list(tree_def), (
        "Deeply nested structure should not be recognized as TreeDef list"
    )

    # Normal list with simple elements (each element has exactly 1 leaf)
    # This is a basic case to confirm the function works correctly
    normal_list = [1, 2]  # Each element contributes 1 leaf
    _, tree_def = tree_flatten(normal_list)
    assert is_treedef_list(tree_def), "Normal list should be TreeDef list"


if __name__ == "__main__":
    test_morph_struct_type()
    test_validate_morph_struct()
    test_create_empty_morph_struct()
    test_var_morph_returns_morphstruct()
    test_var_demorph_with_morphstruct()
    test_empty_var_morph()
    test_all_variables_morph()
    test_is_treedef_list_simple_lists()
    test_is_treedef_list_nested_structures()
    test_is_treedef_list_args_kwargs_structure()
    test_is_treedef_list_non_list_containers()
    test_is_treedef_list_mixed_types()
    test_is_treedef_list_complex_elements()
    test_is_treedef_list_single_value_elements()
    test_is_treedef_list_edge_cases()
    print("All func_utils tests passed!")
