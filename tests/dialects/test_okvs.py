import numpy as np

import mplang.v2.dialects.field as field
import mplang.v2.dialects.tensor as tensor
from mplang.v2.backends.simp_simulator import SimpSimulator
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.edsl import trace


def test_okvs_edsl():
    n = 100
    m = int(n * 1.6)

    # Generate data
    keys_np = np.arange(n, dtype=np.uint64)
    values_np = np.zeros((n, 2), dtype=np.uint64)
    for i in range(n):
        values_np[i, 0] = i
        values_np[i, 1] = i * 10

    host = SimpSimulator(world_size=2)

    def program_fn(keys, values):
        # Solve
        storage = field.solve_okvs(keys, values, m)
        # Decode
        decoded = field.decode_okvs(keys, storage)
        return decoded

    # Run
    # We must treat keys and values as inputs.
    # SimpHost evaluates abstractly?
    # SimpHost executes eagerly if inputs are TensorValue?
    # Actually SimpHost.evaluate_graph traces the function.

    # We can invoke directly if inputs are wrapped?
    # Wait, SimpHost expects a function that takes nothing? Or inputs?
    # Let's use direct invocation since we are testing eager implementation in _mul_impl via run_jax.
    # tensor.run_jax executes eagerly if interpreter is available?
    # No, tensor.run_jax traces if tracing.
    # We want to test EDSL behavior.

    # Trace
    # Use direct TensorValue for runtime inputs (evaluate_graph expects runtime values)
    k_val = TensorValue(keys_np)
    v_val = TensorValue(values_np)

    # For tracing, we need Objects (to infer types/create TraceObjects)
    # tensor.constant creates InterpObjects
    k_obj = tensor.constant(keys_np)
    v_obj = tensor.constant(values_np)

    traced = trace(program_fn, k_obj, v_obj)
    graph = traced.graph

    results = host.evaluate_graph(graph, [k_val, v_val])

    decoded_res = results[0].unwrap()

    assert np.array_equal(decoded_res, values_np), "Decoded values do not match!"
    print("OKVS EDSL Test Passed!")


if __name__ == "__main__":
    test_okvs_edsl()
