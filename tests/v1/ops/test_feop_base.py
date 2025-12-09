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

from mplang.v1.core.mpobject import MPObject
from mplang.v1.core.pfunc import PFunction
from mplang.v1.core.tensor import TensorType
from mplang.v1.ops.base import (
    FeOperation,
    InlineFeOperation,
    SimpleFeOperation,
    get_registry,
    is_feop,
    list_ops,
    stateless_mod,
)
from tests.v1.ops.dummy import DummyTensor


def test_simple_decorator_builds_triad():
    mod = stateless_mod("ut_mod_simple_add")

    @mod.simple_op(pfunc_name="builtin.add")
    def add(
        x: TensorType, y: TensorType, *, alpha: int
    ):  # kernel now receives TensorType specs
        # Return the same spec as first arg
        return x

    assert is_feop(add)
    assert isinstance(add, FeOperation)
    assert isinstance(add, SimpleFeOperation)

    x = DummyTensor(np.float32, (2,))
    y = DummyTensor(np.float32, (2,))

    pfunc, args, out_tree = add(x, y, alpha=1)

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
    assert fetched is add
    ops = list_ops(mod.name)
    assert (mod.name, "add") in ops


def test_feop_decorator_inline():
    mod = stateless_mod("ut_mod_inline_scale")

    @mod.op_def()
    def scale_trace(x: MPObject, factor: int):
        # Build outs info from x type
        leaves, out_tree = tree_flatten(x.mptype._type)
        pfunc = PFunction(
            fn_type="basic.scale",
            ins_info=(x.mptype._type,),
            outs_info=tuple(leaves),
            factor=factor,
        )
        return pfunc, [x], out_tree

    assert is_feop(scale_trace)
    assert isinstance(scale_trace, FeOperation)
    assert isinstance(scale_trace, InlineFeOperation)

    x = DummyTensor(np.float32, (3,))
    pfunc, args, out_tree = scale_trace(x, factor=10)

    assert pfunc.fn_type == "basic.scale"
    assert len(pfunc.ins_info) == 1 and pfunc.ins_info[0] == x.mptype._type
    assert len(pfunc.outs_info) == 1 and pfunc.outs_info[0] == x.mptype._type
    assert pfunc.attrs["factor"] == 10
    assert args == [x]
    assert isinstance(out_tree, PyTreeDef)


def test_simple_rejects_invalid_kwargs():
    mod = stateless_mod("ut_mod_badkw")

    @mod.simple_op(pfunc_name="builtin.echo")
    def echo(x: TensorType):  # returns TensorType
        return x

    # invalid: passing an MPObject as attribute value
    x = DummyTensor(np.int32, (1,))
    with pytest.raises(TypeError):
        echo(x, bad_attr=x)


def test_simple_rejects_invalid_return_type():
    mod = stateless_mod("ut_mod_badret")

    @mod.simple_op(pfunc_name="builtin.bad")
    def bad(
        x: TensorType,
    ):  # missing explicit TensorType/TableType return (returns int instead)
        # Returning a python int should trigger a TypeError in SimpleFeOperation.trace
        return 42

    x = DummyTensor(np.float32, (1,))
    with pytest.raises(TypeError):
        bad(x)
