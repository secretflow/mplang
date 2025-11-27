import jax.numpy as jnp
import numpy as np

import mplang2.backends.tensor_impl  # noqa: F401
from mplang2.dialects import tensor


def test_tensor_ops_e2e():
    """Test basic tensor operations (constant, concat, run_jax) end-to-end."""

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


def test_tensor_elementwise_broadcasting():
    """Test elementwise operation with broadcasting (tensor + scalar)."""

    def workload():
        x = tensor.constant(np.array([10, 20, 30], dtype=np.float32))
        y = tensor.constant(np.array(5, dtype=np.float32))  # Scalar

        def add_fn(a, b):
            return a + b

        z = tensor.elementwise(add_fn, x, y)
        return z

    result = workload()
    result = result.runtime_obj.astype(np.float32)
    expected = np.array([15, 25, 35], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_tensor_elementwise_scalar():
    """Test elementwise operation with pure scalars."""

    def workload():
        x = tensor.constant(np.array(10, dtype=np.float32))
        y = tensor.constant(np.array(20, dtype=np.float32))

        def mul_fn(a, b):
            return a * b

        z = tensor.elementwise(mul_fn, x, y)
        return z

    result = workload()
    # Result should be a 0-d array or scalar
    assert np.ndim(result.runtime_obj) == 0
    assert result.runtime_obj == 200.0
