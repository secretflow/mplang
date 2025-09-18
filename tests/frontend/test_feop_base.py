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

import numpy as np
import pytest
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.cluster import ClusterSpec
from mplang.core.dtype import DType
from mplang.core.mpobject import MPContext, MPObject
from mplang.core.mptype import MPType
from mplang.core.pfunc import PFunction
from mplang.frontend.base import (
    FeOperation,
    InlineFeOperation,
    SimpleFeOperation,
    femod,
    get_registry,
    is_feop,
    list_feops,
)


class DummyContext(MPContext):
    def __init__(self) -> None:
        super().__init__(ClusterSpec.simple(world_size=1))


class DummyTensor(MPObject):
    """Minimal MPObject representing a tensor for frontend tests."""

    def __init__(self, shape: tuple[int, ...], dtype):  # dtype: Any for flexibility
        self._mptype = MPType.tensor(DType.from_any(dtype), shape)
        self._ctx = DummyContext()

    @property
    def mptype(self) -> MPType:
        return self._mptype

    @property
    def ctx(self) -> MPContext:
        return self._ctx


def test_simple_decorator_builds_triad():
    mod = femod("ut_mod_simple_add")

    @mod.simple(name="add", pfunc_name="builtin.add")
    def add_kernel(x: MPObject, y: MPObject, **attrs):  # returns TensorType
        # Return the same type as x
        return x.mptype._type

    assert is_feop(add_kernel)
    assert isinstance(add_kernel, FeOperation)
    assert isinstance(add_kernel, SimpleFeOperation)

    x = DummyTensor((2,), np.float32)
    y = DummyTensor((2,), np.float32)

    pfunc, args, out_tree = add_kernel(x, y, alpha=1)

    assert isinstance(pfunc, PFunction)
    assert pfunc.fn_type == "builtin.add"
    assert len(pfunc.ins_info) == 2
    assert pfunc.ins_info[0] == x.mptype._type
    assert pfunc.ins_info[1] == y.mptype._type
    assert len(pfunc.outs_info) == 1
    assert pfunc.outs_info[0] == x.mptype._type
    assert pfunc.attrs["alpha"] == 1

    assert args == [x, y]
    assert isinstance(out_tree, PyTreeDef)

    # registry checks
    reg = get_registry()
    fetched = reg.get_op(mod.name, "add")
    assert fetched is add_kernel
    ops = list_feops(mod.name)
    assert (mod.name, "add") in ops


def test_feop_decorator_inline():
    mod = femod("ut_mod_inline_scale")

    @mod.feop(name="scale")
    def scale_trace(x: MPObject, factor: int):
        # Build outs info from x type
        leaves, out_tree = tree_flatten(x.mptype._type)
        pfunc = PFunction(
            fn_type="builtin.scale",
            ins_info=(x.mptype._type,),
            outs_info=tuple(leaves),
            factor=factor,
        )
        return pfunc, [x], out_tree

    assert is_feop(scale_trace)
    assert isinstance(scale_trace, FeOperation)
    assert isinstance(scale_trace, InlineFeOperation)

    x = DummyTensor((3,), np.float32)
    pfunc, args, out_tree = scale_trace(x, factor=10)

    assert pfunc.fn_type == "builtin.scale"
    assert len(pfunc.ins_info) == 1 and pfunc.ins_info[0] == x.mptype._type
    assert len(pfunc.outs_info) == 1 and pfunc.outs_info[0] == x.mptype._type
    assert pfunc.attrs["factor"] == 10
    assert args == [x]
    assert isinstance(out_tree, PyTreeDef)


def test_simple_rejects_invalid_kwargs():
    mod = femod("ut_mod_badkw")

    @mod.simple(name="echo", pfunc_name="builtin.echo")
    def echo_kernel(x: MPObject):  # returns TensorType
        return x.mptype._type

    # invalid: passing an MPObject as attribute value
    x = DummyTensor((1,), np.int32)
    with pytest.raises(TypeError):
        echo_kernel(x, bad_attr=x)


def test_simple_rejects_invalid_return_type():
    mod = femod("ut_mod_badret")

    @mod.simple(name="bad", pfunc_name="builtin.bad")
    def bad_kernel(x: MPObject):  # missing explicit TensorType/TableType return
        # Returning a python int should trigger a TypeError in SimpleFeOperation.trace
        return 42

    x = DummyTensor((1,), np.float32)
    with pytest.raises(TypeError):
        bad_kernel(x)
