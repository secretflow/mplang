# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import numpy as np

import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import tensor
from mplang.v2.edsl import serde


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
    np.testing.assert_allclose(result.runtime_obj.unwrap(), expected)


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
    # elementwise_impl returns TensorValue wrapping object array, unwrap it first
    result_val = result.runtime_obj
    if hasattr(result_val, "unwrap"):
        result_val = result_val.unwrap()

    flat = result_val.ravel()
    if flat.size > 0 and hasattr(flat[0], "unwrap"):
        flat = [x.unwrap() for x in flat]

    result = np.array(flat).reshape(result_val.shape).astype(np.float32)
    expected = np.array([13, 24], dtype=np.float32)
    np.testing.assert_allclose(result, expected)
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
    # elementwise_impl returns TensorValue wrapping object array, unwrap it first
    result_val = result.runtime_obj
    if hasattr(result_val, "unwrap"):
        result_val = result_val.unwrap()

    flat = result_val.ravel()
    if flat.size > 0 and hasattr(flat[0], "unwrap"):
        flat = [x.unwrap() for x in flat]

    result = np.array(flat).reshape(result_val.shape).astype(np.float32)
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
    assert np.ndim(result.runtime_obj.unwrap()) == 0
    assert result.runtime_obj.unwrap() == 200.0


# =============================================================================
# Tests: Serialization (Moved from test_serde.py)
# =============================================================================


class TestNumpyArrays:
    """Test serialization of numpy arrays via TensorValue wrapper."""

    def test_1d_array(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        tv = TensorValue(arr)
        result = serde.from_json(serde.to_json(tv))
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.unwrap(), arr)
        assert result.dtype == arr.dtype

    def test_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tv = TensorValue(arr)
        result = serde.from_json(serde.to_json(tv))
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.unwrap(), arr)
        assert result.dtype == arr.dtype

    def test_various_dtypes(self):
        for dtype in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
            arr = np.array([1, 0, 1], dtype=dtype)
            tv = TensorValue(arr)
            result = serde.from_json(serde.to_json(tv))
            assert isinstance(result, TensorValue)
            np.testing.assert_array_equal(result.unwrap(), arr)
            assert result.dtype == arr.dtype

    def test_scalar_array(self):
        arr = np.array(42, dtype=np.int32)
        tv = TensorValue(arr)
        result = serde.from_json(serde.to_json(tv))
        assert isinstance(result, TensorValue)
        np.testing.assert_array_equal(result.unwrap(), arr)

    def test_b64_roundtrip_tensor(self):
        """Test base64 roundtrip with TensorValue."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        data = {"array": TensorValue(arr)}
        b64_str = serde.dumps_b64(data)
        assert isinstance(b64_str, str)
        result = serde.loads_b64(b64_str)
        assert isinstance(result["array"], TensorValue)
        np.testing.assert_array_equal(result["array"].unwrap(), arr)
