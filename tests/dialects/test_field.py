import numpy as np

from mplang.v2.backends.simp_simulator import SimpSimulator
from mplang.v2.dialects import field, tensor
from mplang.v2.edsl import trace


def test_field_mul_integration():
    """Verify field.mul invokes C++ kernel correctly."""
    sim = SimpSimulator(world_size=1)

    def protocol():
        # Create Inputs (uint64 pairs representing GF128 elements)
        # A = [1, 2] (low, high)
        # B = [3, 4]
        a = tensor.constant(np.array([1, 2], dtype=np.uint64))
        b = tensor.constant(np.array([3, 4], dtype=np.uint64))

        # Use field.mul
        res = field.mul(a, b)
        return res

    traced = trace(protocol)
    print("Graph:", traced.graph.to_string())

    # Execute
    res = sim.evaluate_graph(traced.graph, [])
    val = res[0].unwrap()

    print("Result:", val)

    # Expected: [1083, 2]
    expected = np.array([1083, 2], dtype=np.uint64)
    assert np.array_equal(val, expected), f"Expected {expected}, got {val}"

    sim.shutdown()


if __name__ == "__main__":
    test_field_mul_integration()
