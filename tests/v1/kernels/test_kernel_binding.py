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

# Tests for per-RuntimeContext binding isolation
from __future__ import annotations

import numpy as np
import pytest

from mplang.v1.core.dtypes import (
    INT64,  # switched from INT32 to INT64 to match Python int
)
from mplang.v1.core.pfunc import PFunction
from mplang.v1.core.tensor import TensorType
from mplang.v1.kernels import base
from mplang.v1.kernels.context import RuntimeContext
from mplang.v1.kernels.value import TensorValue

# We'll register two fake kernels for an op to test rebinding.
# If they already exist due to other tests, we guard with try/except.


@base.kernel_def("test.echo.v1")
def _echo_v1(
    pfunc: PFunction, x: TensorValue
) -> tuple[TensorValue,]:  # pragma: no cover - executed in test
    arr = x.to_numpy()
    result = np.asarray(arr + 1, dtype=arr.dtype)
    return (TensorValue(result),)


@base.kernel_def("test.echo.v2")
def _echo_v2(
    pfunc: PFunction, x: TensorValue
) -> tuple[TensorValue,]:  # pragma: no cover - executed in test
    arr = x.to_numpy()
    result = np.asarray(arr + 2, dtype=arr.dtype)
    return (TensorValue(result),)


def make_pfunc(op_type: str) -> PFunction:
    # Minimal PFunction stub compatible with backend.run_kernel expectations.
    # shape info matters only for validation; use scalar INT64 (Python int maps to int64).
    return PFunction(
        fn_type=op_type,
        fn_text="",
        ins_info=[TensorType(shape=(), dtype=INT64)],
        outs_info=[TensorType(shape=(), dtype=INT64)],
    )


def test_isolated_rebind():
    # ctx1 binds op -> v1, ctx2 binds op -> v2; they should not interfere.
    op = "test.echo"
    ctx1 = RuntimeContext(rank=0, world_size=1, initial_bindings={op: "test.echo.v1"})
    ctx2 = RuntimeContext(rank=0, world_size=1, initial_bindings={op: "test.echo.v2"})

    pfunc = make_pfunc(op)
    out1 = ctx1.run_kernel(pfunc, [TensorValue(np.array(10, dtype=np.int64))])[0]
    out2 = ctx2.run_kernel(pfunc, [TensorValue(np.array(10, dtype=np.int64))])[0]

    assert out1.to_numpy().item() == 11
    assert out2.to_numpy().item() == 12


def test_rebind_only_affects_context():
    op = "test.echo"
    ctx = RuntimeContext(rank=0, world_size=1, initial_bindings={op: "test.echo.v1"})
    pfunc = make_pfunc(op)
    assert (
        ctx.run_kernel(pfunc, [TensorValue(np.array(5, dtype=np.int64))])[0]
        .to_numpy()
        .item()
        == 6
    )
    ctx.rebind_op(op, "test.echo.v2")
    assert (
        ctx.run_kernel(pfunc, [TensorValue(np.array(5, dtype=np.int64))])[0]
        .to_numpy()
        .item()
        == 7
    )


def test_force_flag():
    op = "test.echo"
    ctx = RuntimeContext(rank=0, world_size=1, initial_bindings={op: "test.echo.v1"})
    # Attempt non-force bind (should keep v1)
    ctx.bind_op(op, "test.echo.v2", force=False)
    pfunc = make_pfunc(op)
    assert (
        ctx.run_kernel(pfunc, [TensorValue(np.array(1, dtype=np.int64))])[0]
        .to_numpy()
        .item()
        == 2
    )  # still v1 (+1)
    # Now force
    ctx.bind_op(op, "test.echo.v2", force=True)
    assert (
        ctx.run_kernel(pfunc, [TensorValue(np.array(1, dtype=np.int64))])[0]
        .to_numpy()
        .item()
        == 3
    )


def test_unknown_kernel_id():
    ctx = RuntimeContext(rank=0, world_size=1)
    with pytest.raises(KeyError):
        ctx.bind_op("some.op", "non.existent.kernel")


def test_missing_binding():
    # Pick an op name unlikely in defaults
    op = "unit.test.unbound"
    ctx = RuntimeContext(rank=0, world_size=1)
    pfunc = make_pfunc(op)
    with pytest.raises(NotImplementedError):
        ctx.run_kernel(pfunc, [TensorValue(np.array(0, dtype=np.int64))])
