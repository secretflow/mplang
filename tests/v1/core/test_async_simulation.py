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

import jax.numpy as jnp
import numpy as np
import pytest

from mplang.v1.core import ClusterSpec, Mask
from mplang.v1.core.context_mgr import with_ctx
from mplang.v1.core.primitive import function
from mplang.v1.core.tracer import TraceContext, trace
from mplang.v1.ops import jax_cc
from mplang.v1.runtime.simulation import Simulator, SimVar
from mplang.v1.simp.api import constant, run


def add(x, y):
    return run(None, jax_cc.run_jax, lambda a, b: jnp.add(a, b), x, y)


@pytest.mark.asyncio
async def test_async_simulation_basic():
    """Test basic async simulation."""

    # 1. Define a function
    @function
    def add_func(x, y):
        return add(x, y)

    # 2. Trace it
    cluster = ClusterSpec.simple(2)
    ctx = TraceContext(cluster, mask=Mask(3))

    # Let's trace a function that adds two constants.
    @function
    def simple_add():
        a = constant(10)
        b = constant(20)
        return add(a, b)

    traced = trace(ctx, simple_add)
    expr = traced.make_expr()

    # 3. Create Simulator
    sim = Simulator(cluster)

    # 4. Evaluate Async
    # simple_add takes no args, so bindings is empty
    # make_expr() returns a FuncDefExpr. We want to evaluate its body.
    results = await sim._evaluate_async(expr.body, {})

    # 5. Verify
    assert len(results) == 1
    sim_var = results[0]
    assert isinstance(sim_var, SimVar)
    values = sim_var.values
    assert len(values) == 2
    assert values[0] == 30
    assert values[1] == 30


@pytest.mark.asyncio
async def test_async_simulation_args():
    """Test async simulation with arguments."""

    @function
    def add_func(x, y):
        return add(x, y)

    cluster = ClusterSpec.simple(2)
    ctx = TraceContext(cluster, mask=Mask(3))

    # Let's manually create TraceVars for inputs.
    from mplang.v1.core.dtypes import INT32
    from mplang.v1.core.expr.ast import VariableExpr
    from mplang.v1.core.mptype import MPType
    from mplang.v1.core.tracer import TraceVar
    from mplang.v1.kernels.value import TensorValue

    x_type = MPType.tensor(INT32, (), Mask(3))
    y_type = MPType.tensor(INT32, (), Mask(3))

    with with_ctx(ctx):
        x = TraceVar(ctx, VariableExpr("x", x_type))
        y = TraceVar(ctx, VariableExpr("y", y_type))

    traced = trace(ctx, add_func, x, y)
    expr = traced.make_expr()

    # Now evaluate with bindings
    sim = Simulator(cluster)

    # Create input values
    x_val = SimVar(
        sim,
        x_type,
        [
            TensorValue(np.array(10, dtype=np.int32)),
            TensorValue(np.array(10, dtype=np.int32)),
        ],
    )
    y_val = SimVar(
        sim,
        y_type,
        [
            TensorValue(np.array(20, dtype=np.int32)),
            TensorValue(np.array(20, dtype=np.int32)),
        ],
    )

    bindings = {"x": x_val, "y": y_val}
    # expr is FuncDefExpr. We need to evaluate its body.
    # But wait, the body refers to parameters.
    # FuncDefExpr params are generated names usually?
    # Or they match the names we gave?

    # When we use `trace(ctx, func, x, y)`, `x` and `y` are passed as arguments.
    # `trace` captures them.

    # If `x` and `y` are TraceVars with VariableExpr, they are treated as inputs?
    # `trace` logic:
    # It calls the function with arguments.
    # If arguments are TraceVars, they are used.

    # The resulting FuncDefExpr will have parameters corresponding to the inputs.
    # But `x` and `y` are captured from the outer scope if we pass them?
    # No, they are passed as arguments.

    # Let's check `traced.make_expr()` logic.
    # It creates a FuncDefExpr.
    # The parameters of FuncDefExpr correspond to the arguments of the traced function.

    # But we need to know the parameter names to bind them.
    # `traced.make_expr()` might generate parameter names.

    # Let's inspect `expr.params`.

    # For now, let's assume we can bind by position if we construct a CallExpr.
    # But `evaluate` takes `bindings` which is a dict.

    # If we evaluate `expr.body`, it contains `VariableExpr`s.
    # These `VariableExpr`s refer to the parameter names.

    # So we need to map our input values to these parameter names.

    # `traced.make_expr()` returns `FuncDefExpr`.
    # `expr.params` gives the list of parameter names.

    # So we should map `expr.params` to our values.

    param_names = expr.params
    assert len(param_names) == 2

    bindings = {param_names[0]: x_val, param_names[1]: y_val}

    results = await sim._evaluate_async(expr.body, bindings)

    assert len(results) == 1
    assert results[0].values == [30, 30]


def test_sync_evaluate():
    """Test synchronous evaluate."""

    @function
    def simple_add():
        a = constant(10)
        b = constant(20)
        return add(a, b)

    cluster = ClusterSpec.simple(2)
    ctx = TraceContext(cluster, mask=Mask(3))
    traced = trace(ctx, simple_add)
    expr = traced.make_expr()

    sim = Simulator(cluster)
    # This calls evaluate (sync) which calls asyncio.run(_evaluate_async)
    results = sim.evaluate(expr.body, {})

    assert len(results) == 1
    values = results[0].values
    assert values[0] == 30
    assert values[1] == 30


@pytest.mark.asyncio
async def test_evaluate_in_loop():
    """Test evaluate inside an async loop (should fail without nest_asyncio or work with it)."""

    @function
    def simple_add():
        a = constant(10)
        b = constant(20)
        return add(a, b)

    cluster = ClusterSpec.simple(2)
    ctx = TraceContext(cluster, mask=Mask(3))
    traced = trace(ctx, simple_add)
    expr = traced.make_expr()

    sim = Simulator(cluster)

    # We are in an async test, so there is a running loop.
    # Calling sim.evaluate() should raise RuntimeError or work if nest_asyncio is present.
    import asyncio

    await asyncio.sleep(0)  # Silence RUF029 and ensure loop is running

    import importlib.util

    nest_asyncio_installed = importlib.util.find_spec("nest_asyncio") is not None

    if nest_asyncio_installed:
        # If nest_asyncio is installed, it should work (assuming apply() is called inside evaluate)
        results = sim.evaluate(expr.body, {})
        assert len(results) == 1
        values = results[0].values
        assert values[0] == 30
    else:
        # If not installed, it should raise RuntimeError
        with pytest.raises(RuntimeError, match="nest_asyncio"):
            sim.evaluate(expr.body, {})
