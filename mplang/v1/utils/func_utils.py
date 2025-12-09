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

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from jax.tree_util import PyTreeDef, tree_flatten, tree_unflatten

# Type alias for the structure information returned by var_morph
# This represents (values_tree, split_info) where:
# - values_tree: PyTreeDef from JAX tree_flatten
# - split_info: tuple of indices where variables were located in the flattened list
# This type is never None - var_morph always returns a valid MorphStruct
MorphStruct = tuple[PyTreeDef, tuple[int, ...]]


def validate_morph_struct(morph_struct: MorphStruct) -> None:
    """Validate that a MorphStruct has the correct structure.

    Args:
        morph_struct: The MorphStruct to validate

    Raises:
        TypeError: If the MorphStruct has invalid structure or types
    """
    if not isinstance(morph_struct, tuple) or len(morph_struct) != 2:
        raise TypeError(
            f"MorphStruct must be a tuple of length 2, got {type(morph_struct)}"
        )

    values_tree, split_info = morph_struct
    if not isinstance(values_tree, PyTreeDef):
        raise TypeError(f"MorphStruct[0] must be PyTreeDef, got {type(values_tree)}")

    if not isinstance(split_info, tuple):
        raise TypeError(
            f"MorphStruct[1] must be tuple[int, ...], got {type(split_info)}"
        )

    if not all(isinstance(x, int) for x in split_info):
        raise TypeError(
            f"MorphStruct[1] must contain only integers, got types: {[type(x) for x in split_info]}"
        )


def list_split(origin: list, pred: Callable) -> tuple[list, list, list]:
    fst, snd = [], []
    fst_idxs = []

    for idx, x in enumerate(origin):
        if pred(x):
            fst.append(x)
            fst_idxs.append(idx)
        else:
            snd.append(x)

    return fst, snd, fst_idxs


def list_reconstruct(fst: list, snd: list, fst_idxs: list) -> list:
    result = []
    fst_itr, snd_itr = 0, 0
    for idx in range(len(fst) + len(snd)):
        if fst_itr < len(fst_idxs) and fst_idxs[fst_itr] == idx:
            result.append(fst[fst_itr])
            fst_itr += 1
        else:
            result.append(snd[snd_itr])
            snd_itr += 1

    return result


def var_morph(
    values: Any, is_variable: Callable[[Any], bool]
) -> tuple[list, list, MorphStruct]:
    """aka. flat_and_split Flat and split variable from immediates"""
    values_flat, values_tree = tree_flatten(values)
    variables, immediates, split_info = list_split(values_flat, is_variable)
    return variables, immediates, (values_tree, tuple(split_info))


def var_demorph(variables: list, immediates: list, morph_info: MorphStruct) -> Any:
    """aka. merge_and_unflat. Merge vars and immediates, and reconstruct the tree."""
    values_tree, split_info = morph_info
    values_flat = list_reconstruct(variables, immediates, list(split_info))
    return tree_unflatten(values_tree, values_flat)


def normalize_fn(
    fn: Callable, args: Any, kwargs: Any, is_variable: Callable[[Any], bool]
) -> tuple[Callable, list]:
    """Flatten (args, kwargs) and capture immediate.
    Returns the a function captures all immediates and a list of variables.
    """
    params, immediates, morph = var_morph((args, kwargs), is_variable)

    @functools.wraps(fn)
    def normalized(rargs: list[Any]) -> tuple[Any]:
        # aargs is short actual arguments.
        # reconstruct original paramter, replace the traced object with actual object.
        aargs, akwargs = var_demorph(rargs, immediates, morph)
        return fn(*aargs, **akwargs)  # type: ignore[no-any-return]

    return normalized, params


def is_treedef_list(treedef: PyTreeDef) -> bool:
    """
    Checks if a PyTreeDef object represents a simple (non-nested) list.

    A 'simple list' PyTreeDef must satisfy two conditions:
    1. Its root container type is `list`.
    2. All of its children are leaf nodes (each child has exactly one value).

    Args:
        treedef: The PyTreeDef object to check.

    Returns:
        True if the treedef represents a simple list, False otherwise.
    """
    # 1. Check if the root node type is a list
    node_data = treedef.node_data()
    if node_data is None or node_data[0] is not list:
        return False

    # 2. Get the PyTreeDefs of all children
    children_treedefs = treedef.children()

    # 3. Check if all children are leaf nodes.
    #    all() returns True for an empty iterable, which is the correct
    #    behavior for an empty list [].
    return all(child.num_leaves == 1 for child in children_treedefs)
