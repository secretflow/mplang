# Tests for per-RuntimeContext binding isolation
from __future__ import annotations

import pytest

from mplang.backend import base
from mplang.backend.context import RuntimeContext
from mplang.core.dtype import INT64  # switched from INT32 to INT64 to match Python int
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType

# We'll register two fake kernels for an op to test rebinding.
# If they already exist due to other tests, we guard with try/except.


@base.kernel_def("test.echo.v1")
def _echo_v1(pfunc, x):  # pragma: no cover - executed in test
    return (x + 1,)


@base.kernel_def("test.echo.v2")
def _echo_v2(pfunc, x):  # pragma: no cover - executed in test
    return (x + 2,)


def make_pfunc(op_type: str) -> PFunction:
    # Minimal PFunction stub compatible with backend.run_kernel expectations.
    # shape info matters only for validation; use scalar INT64 (python int maps to int64).
    return PFunction(
        fn_type=op_type,
        fn_text="",
        ins_info=[TensorType(shape=(), dtype=INT64)],
        outs_info=[TensorType(shape=(), dtype=INT64)],
    )


def test_isolated_rebind():
    # ctx1 binds op -> v1, ctx2 binds op -> v2; they should not interfere.
    op = "test.echo"
    ctx1 = RuntimeContext(rank=0, world_size=1, bindings={op: "test.echo.v1"})
    ctx2 = RuntimeContext(rank=0, world_size=1, bindings={op: "test.echo.v2"})

    pfunc = make_pfunc(op)
    out1 = ctx1.run_kernel(pfunc, [10])[0]
    out2 = ctx2.run_kernel(pfunc, [10])[0]

    assert out1 == 11
    assert out2 == 12


def test_rebind_only_affects_context():
    op = "test.echo"
    ctx = RuntimeContext(rank=0, world_size=1, bindings={op: "test.echo.v1"})
    pfunc = make_pfunc(op)
    assert ctx.run_kernel(pfunc, [5])[0] == 6
    ctx.rebind_op(op, "test.echo.v2")
    assert ctx.run_kernel(pfunc, [5])[0] == 7


def test_force_flag():
    op = "test.echo"
    ctx = RuntimeContext(rank=0, world_size=1, bindings={op: "test.echo.v1"})
    # Attempt non-force bind (should keep v1)
    ctx.bind_op(op, "test.echo.v2", force=False)
    pfunc = make_pfunc(op)
    assert ctx.run_kernel(pfunc, [1])[0] == 2  # still v1 (+1)
    # Now force
    ctx.bind_op(op, "test.echo.v2", force=True)
    assert ctx.run_kernel(pfunc, [1])[0] == 3


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
        ctx.run_kernel(pfunc, [0])
