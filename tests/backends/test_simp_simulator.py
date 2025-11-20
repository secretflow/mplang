import jax.numpy as jnp

# Register runtimes
import mplang2.backends.tensor_impl  # noqa: F401
from mplang2.backends.simp_host import HostVar
from mplang2.backends.simp_simulator import SimpSimulator, get_or_create_context
from mplang2.dialects import simp
from mplang2.dialects.simp import pcall_static, uniform_cond
from mplang2.dialects.tensor import run_jax
from mplang2.edsl.interpreter import InterpObject
from mplang2.edsl.typing import MPType, Tensor, f32, i64


def add(x, y):
    return run_jax(jnp.add, x, y)


def mul(x, y):
    return run_jax(jnp.multiply, x, y)


def test_pcall_static():
    # Define a function to run on parties
    def my_func(x, y):
        return add(x, y)

    # Create interpreter
    interp = SimpSimulator(world_size=3)

    # Create inputs (HostVars)
    # World size 3
    get_or_create_context(3)

    x_val = HostVar([1, 2, 3])
    y_val = HostVar([10, 20, 30])

    t_type = MPType(Tensor[f32, ()], parties=(0, 1, 2))
    x_obj = InterpObject(x_val, t_type, interp)
    y_obj = InterpObject(y_val, t_type, interp)

    # Call pcall_static
    with interp:
        res = pcall_static((0, 1, 2), my_func, x_obj, y_obj)

    assert isinstance(res, InterpObject)
    assert isinstance(res.runtime_obj, HostVar)
    # Note: run_jax returns numpy arrays (or jax arrays), so we compare values
    # HostVar holds list of values.
    # 1+10=11, 2+20=22, 3+30=33
    assert res.runtime_obj.values == [11, 22, 33]


def test_uniform_cond():
    interp = SimpSimulator(world_size=2)
    get_or_create_context(2)

    # True condition
    pred_true = InterpObject(HostVar([True, True]), MPType(i64, (0, 1)), interp)

    x_val = HostVar([1, 2])
    x_obj = InterpObject(x_val, MPType(Tensor[f32, ()], parties=(0, 1)), interp)

    def then_fn(x):
        return pcall_static((0, 1), lambda a: add(a, a), x)

    def else_fn(x):
        return pcall_static((0, 1), lambda a: mul(a, a), x)

    with interp:
        res = uniform_cond(pred_true, then_fn, else_fn, x_obj)

    assert res.runtime_obj.values == [2, 4]

    # Test False case
    pred_val_false = HostVar([False, False])
    pred_obj_false = InterpObject(pred_val_false, MPType(i64, (0, 1)), interp)

    with interp:
        res_false = uniform_cond(pred_obj_false, then_fn, else_fn, x_obj)
    assert res_false.runtime_obj.values == [1, 4]


def test_while_loop_eager():
    """Test simp.while_loop eager execution."""

    def cond(val):
        # val is TraceObject during tracing
        def local_cond(x):
            return run_jax(lambda a: a < 10, x)

        return pcall_static((0, 1), local_cond, val)

    def body(val):
        def local_body(x):
            return run_jax(lambda a: a + 1, x)

        return pcall_static((0, 1), local_body, val)

    # Setup runtime
    # Reset global context to ensure world_size=2
    import mplang2.backends.simp_simulator as simp_runtime

    if simp_runtime._SIM_CONTEXT:
        simp_runtime._SIM_CONTEXT.shutdown()
        simp_runtime._SIM_CONTEXT = None

    get_or_create_context(world_size=2)
    interp = SimpSimulator(world_size=2)

    start_val = HostVar([0, 0])
    t_type = MPType(Tensor[i64, ()], parties=(0, 1))
    start_obj = InterpObject(start_val, t_type, interp)

    # Eager call
    with interp:
        res = simp.while_loop(cond, body, start_obj)

    assert isinstance(res, InterpObject)
    assert res.runtime_obj.values == [10, 10]
