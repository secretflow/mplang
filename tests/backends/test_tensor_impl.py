import jax.numpy as jnp
import numpy as np

import mplang2.backends.tensor_impl  # noqa: F401
from mplang2.dialects import tensor
from mplang2.edsl.jit import jit


def test_tensor_ops_e2e():
    """Test basic tensor operations (constant, concat, run_jax) end-to-end."""

    @jit
    def workload():
        # Create constants
        x = tensor.constant(np.array([1, 2, 3], dtype=np.float32))
        y = tensor.constant(np.array([4, 5, 6], dtype=np.float32))

        # Test concat
        # x: [1, 2, 3], y: [4, 5, 6] -> z: [1, 2, 3, 4, 5, 6]
        z = tensor.concat([x, y], axis=0)

        # Test run_jax
        def square_fn(a):
            return jnp.square(a)

        # w: [1, 4, 9, 16, 25, 36]
        w = tensor.run_jax(square_fn, z)
        return w

    # Execute
    result = workload()

    # Verify
    expected = np.array([1, 4, 9, 16, 25, 36], dtype=np.float32)
    np.testing.assert_allclose(result.runtime_obj, expected)


def test_tensor_elementwise():
    """Test elementwise operation."""

    @jit
    def workload():
        x = tensor.constant(np.array([10, 20], dtype=np.float32))
        y = tensor.constant(np.array([3, 4], dtype=np.float32))

        # Elementwise add
        def add_fn(a, b):
            return a + b

        z = tensor.elementwise(add_fn, x, y)
        return z

    result = workload()
    # elementwise_impl returns object array, cast to float for comparison
    result = result.runtime_obj.astype(np.float32)
    expected = np.array([13, 24], dtype=np.float32)
    np.testing.assert_allclose(result, expected)
