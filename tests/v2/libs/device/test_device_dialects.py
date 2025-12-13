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

"""Tests for device API with various input types and dialects.

This module tests:
- put() with non-JAX objects (numpy, Python scalars, lists, etc.)
- Device API working with different dialects (BFV, table, etc.)
- Cross-dialect operations on devices
"""

import numpy as np
import pytest

import mplang.v2.backends.bfv_impl  # noqa: F401
import mplang.v2.backends.table_impl  # noqa: F401
import mplang.v2.backends.tensor_impl  # noqa: F401 - Register backend
from mplang.v2.dialects import bfv, table, tensor
from mplang.v2.libs.device import device, get_dev_attr, put


def extract_runtime_value(obj):
    """Extract runtime value from Object.

    In multi-party simulation, runtime_obj may be a DriverVar containing
    values per party. This helper extracts the first non-None value.
    """
    from mplang.v2 import fetch

    val = fetch(obj)

    # If result is a list (one per party), return first non-None
    if isinstance(val, list):
        for v in val:
            if v is not None:
                return v
    return val


# =============================================================================
# put() with Non-JAX Objects
# =============================================================================


class TestPutNonJaxObjects:
    """Test put() with various non-JAX input types."""

    def test_put_numpy_array(self, ctx_3pc):
        """put() should work with numpy arrays."""
        x = np.array([1, 2, 3], dtype=np.int64)
        x_dev = put("P0", x)

        assert get_dev_attr(x_dev) == "P0"

    def test_put_numpy_float_array(self, ctx_3pc):
        """put() should work with numpy float arrays."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x_dev = put("P0", x)

        assert get_dev_attr(x_dev) == "P0"

    def test_put_numpy_scalar(self, ctx_3pc):
        """put() should work with numpy scalars."""
        x = np.int64(42)
        x_dev = put("P0", x)

        assert get_dev_attr(x_dev) == "P0"

    def test_put_python_int(self, ctx_3pc):
        """put() should work with Python integers."""
        x = 42
        x_dev = put("P0", x)

        assert get_dev_attr(x_dev) == "P0"

    def test_put_python_float(self, ctx_3pc):
        """put() should work with Python floats."""
        x = 3.14
        x_dev = put("P0", x)

        assert get_dev_attr(x_dev) == "P0"

    def test_put_python_list(self, ctx_3pc):
        """put() should work with Python lists."""
        x = [1, 2, 3]
        x_dev = put("P0", x)

        assert get_dev_attr(x_dev) == "P0"

    def test_put_nested_list(self, ctx_3pc):
        """put() should work with nested Python lists."""
        x = [[1, 2], [3, 4]]
        x_dev = put("P0", x)

        assert get_dev_attr(x_dev) == "P0"

    def test_put_numpy_2d_array(self, ctx_3pc):
        """put() should work with 2D numpy arrays."""
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        x_dev = put("P0", x)

        assert get_dev_attr(x_dev) == "P0"


# =============================================================================
# Device API with Tensor Dialect
# =============================================================================


class TestDeviceWithTensorDialect:
    """Test device API with tensor dialect operations."""

    def test_tensor_constant_on_device(self, ctx_3pc):
        """tensor.constant inside device function should work."""

        @device("P0")
        def create_tensor():
            return tensor.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))

        result = create_tensor()
        assert get_dev_attr(result) == "P0"

    def test_tensor_run_jax_on_device(self, ctx_3pc):
        """tensor.run_jax inside device function should work."""

        x = put("P0", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = put("P0", np.array([4.0, 5.0, 6.0], dtype=np.float32))

        @device("P0")
        def add_tensors(a, b):
            return tensor.run_jax(lambda x, y: x + y, a, b)

        result = add_tensors(x, y)
        assert get_dev_attr(result) == "P0"


# =============================================================================
# Device API with BFV Dialect
# =============================================================================


class TestDeviceWithBFVDialect:
    """Test device API with BFV (FHE) dialect operations."""

    def test_bfv_keygen_on_device(self, ctx_3pc):
        """BFV keygen should work on PPU device."""

        @device("P0")
        def gen_keys():
            pk, sk = bfv.keygen(poly_modulus_degree=4096)
            return pk, sk

        pk, sk = gen_keys()
        assert get_dev_attr(pk) == "P0"
        assert get_dev_attr(sk) == "P0"

    def test_bfv_encrypt_decrypt_on_device(self, ctx_3pc):
        """BFV encrypt/decrypt cycle should work on PPU device."""

        @device("P0")
        def bfv_cycle():
            # Setup
            pk, sk = bfv.keygen(poly_modulus_degree=4096)
            encoder = bfv.create_encoder(poly_modulus_degree=4096)

            # Data
            v = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))

            # Encode & Encrypt
            pt = bfv.encode(v, encoder)
            ct = bfv.encrypt(pt, pk)

            # Decrypt & Decode
            pt_dec = bfv.decrypt(ct, sk)
            result = bfv.decode(pt_dec, encoder)

            return result

        result = bfv_cycle()
        assert get_dev_attr(result) == "P0"

        # Extract actual value from simulation result
        val = extract_runtime_value(result)
        np.testing.assert_array_equal(val[:4], [1, 2, 3, 4])

    def test_bfv_homomorphic_add_on_device(self, ctx_3pc):
        """BFV homomorphic addition should work on PPU device."""

        @device("P0")
        def bfv_add():
            pk, sk = bfv.keygen(poly_modulus_degree=4096)
            encoder = bfv.create_encoder(poly_modulus_degree=4096)

            v1 = tensor.constant(np.array([1, 2, 3, 4], dtype=np.int64))
            v2 = tensor.constant(np.array([10, 20, 30, 40], dtype=np.int64))

            pt1 = bfv.encode(v1, encoder)
            ct1 = bfv.encrypt(pt1, pk)

            pt2 = bfv.encode(v2, encoder)
            ct2 = bfv.encrypt(pt2, pk)

            # Homomorphic addition
            ct_sum = bfv.add(ct1, ct2)

            # Decrypt
            pt_sum = bfv.decrypt(ct_sum, sk)
            return bfv.decode(pt_sum, encoder)

        result = bfv_add()
        assert get_dev_attr(result) == "P0"

        # Extract actual value from simulation result
        val = extract_runtime_value(result)
        np.testing.assert_array_equal(val[:4], [11, 22, 33, 44])


# =============================================================================
# Device API with Table Dialect
# =============================================================================


class TestDeviceWithTableDialect:
    """Test device API with table (SQL) dialect operations."""

    def test_table_constant_on_device(self, ctx_3pc):
        """table.constant should work on PPU device."""

        @device("P0")
        def create_table():
            data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
            return table.constant(data)

        result = create_table()
        assert get_dev_attr(result) == "P0"

    def test_table_sql_on_device(self, ctx_3pc):
        """table.run_sql should work on PPU device."""
        import mplang.v2.edsl.typing as elt

        @device("P0")
        def run_sql_query():
            data = {"x": [10, 20, 30], "y": [1, 2, 3]}
            t = table.constant(data)

            # Run SQL to compute x + y
            out_schema = elt.TableType({
                "sum": elt.TensorType(elt.i64, ()),
            })
            result = table.run_sql(
                "SELECT x + y AS sum FROM t", out_type=out_schema, t=t
            )
            return result

        result = run_sql_query()
        assert get_dev_attr(result) == "P0"


# =============================================================================
# Cross-Dialect Operations on Device
# =============================================================================


class TestCrossDialectOnDevice:
    """Test operations that combine multiple dialects on a device."""

    def test_tensor_to_table_conversion_on_device(self, ctx_3pc):
        """Converting tensor to table should work on device."""

        @device("P0")
        def tensor_to_table():
            # Create a tensor
            t = tensor.constant(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64))

            # Convert to table
            tbl = table.tensor2table(t, column_names=["a", "b"])
            return tbl

        result = tensor_to_table()
        assert get_dev_attr(result) == "P0"

    def test_table_to_tensor_conversion_on_device(self, ctx_3pc):
        """Converting table to tensor should work on device."""

        @device("P0")
        def table_to_tensor():
            # Create a table
            data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
            tbl = table.constant(data)

            # Convert to tensor
            t = table.table2tensor(tbl, number_rows=3)
            return t

        result = table_to_tensor()
        assert get_dev_attr(result) == "P0"


# =============================================================================
# Device Transfer with Non-JAX Objects
# =============================================================================


class TestDeviceTransferNonJax:
    """Test device-to-device transfer with non-JAX data."""

    def test_transfer_numpy_between_ppus(self, ctx_3pc):
        """Transfer numpy array between PPU devices."""
        x = np.array([1, 2, 3], dtype=np.int64)
        x_p0 = put("P0", x)
        x_p1 = put("P1", x_p0)

        assert get_dev_attr(x_p0) == "P0"
        assert get_dev_attr(x_p1) == "P1"

    def test_transfer_numpy_ppu_to_spu(self, ctx_3pc):
        """Transfer numpy array from PPU to SPU."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x_p0 = put("P0", x)
        x_sp0 = put("SP0", x_p0)

        assert get_dev_attr(x_sp0) == "SP0"

    def test_transfer_numpy_spu_to_ppu(self, ctx_3pc):
        """Transfer numpy array from SPU to PPU (reveal)."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x_sp0 = put("SP0", x)
        x_p0 = put("P0", x_sp0)

        assert get_dev_attr(x_p0) == "P0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
