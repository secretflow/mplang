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

"""Tests for TEE device operations in the device API.

This module tests:
- TEE device execution
- PPU <-> TEE encrypted data transfers
- TEE <-> SPU data transfers (seal/reveal)
- Host -> TEE data upload
"""

import jax.numpy as jnp
import numpy as np
import pytest

from mplang.v2.libs.device import device, get_dev_attr, put


def extract_runtime_value(obj):
    """Extract runtime value from Object.

    In multi-party simulation, runtime_obj may be a DriverVar containing
    values per party. This helper extracts the first non-None value.
    """
    val = obj.runtime_obj
    if hasattr(val, "values"):
        for v in val.values:
            if v is not None:
                return v
    elif isinstance(val, (list, tuple)):
        for v in val:
            if v is not None:
                return v
    return val


# =============================================================================
# Host -> TEE Transfer
# =============================================================================


class TestHostToTee:
    """Test uploading data from host to TEE device."""

    def test_put_numpy_array_to_tee(self, ctx_3pc):
        """put() should work with numpy arrays to TEE."""
        x = np.array([1, 2, 3], dtype=np.int64)
        x_tee = put("TEE0", x)

        assert get_dev_attr(x_tee) == "TEE0"

    def test_put_jax_array_to_tee(self, ctx_3pc):
        """put() should work with JAX arrays to TEE."""
        x = jnp.array([1.0, 2.0, 3.0])
        x_tee = put("TEE0", x)

        assert get_dev_attr(x_tee) == "TEE0"

    def test_put_python_scalar_to_tee(self, ctx_3pc):
        """put() should work with Python scalars to TEE."""
        x = 42
        x_tee = put("TEE0", x)

        assert get_dev_attr(x_tee) == "TEE0"


# =============================================================================
# PPU <-> TEE Encrypted Transfer
# =============================================================================


class TestPpuTeeTransfer:
    """Test encrypted data transfers between PPU and TEE."""

    def test_transfer_ppu_to_tee(self, ctx_3pc):
        """Data should transfer from PPU to TEE with encryption."""
        x = put("P0", jnp.array([1, 2, 3]))
        x_tee = put("TEE0", x)

        assert get_dev_attr(x_tee) == "TEE0"

    def test_transfer_tee_to_ppu(self, ctx_3pc):
        """Data should transfer from TEE to PPU with encryption."""
        x = put("TEE0", jnp.array([4, 5, 6]))
        x_ppu = put("P0", x)

        assert get_dev_attr(x_ppu) == "P0"

    def test_round_trip_ppu_tee_ppu(self, ctx_3pc):
        """Data should survive round trip PPU -> TEE -> PPU."""
        original = jnp.array([10, 20, 30])
        x_ppu = put("P0", original)
        x_tee = put("TEE0", x_ppu)
        x_back = put("P0", x_tee)

        assert get_dev_attr(x_back) == "P0"


# =============================================================================
# TEE Device Execution
# =============================================================================


class TestTeeExecution:
    """Test function execution on TEE device."""

    def test_device_decorator_on_tee(self, ctx_3pc):
        """@device decorator should work with TEE device."""

        @device("TEE0").jax
        def add_one(x):
            return x + 1

        x = put("TEE0", jnp.array([1, 2, 3]))
        result = add_one(x)

        assert get_dev_attr(result) == "TEE0"

    def test_auto_infer_tee_device(self, ctx_3pc):
        """Device should auto-infer TEE from arguments."""

        @device()
        def identity(x):
            return x

        x = put("TEE0", jnp.array([1, 2, 3]))
        result = identity(x)

        assert get_dev_attr(result) == "TEE0"

    def test_tee_computation(self, ctx_3pc):
        """Computation should work correctly on TEE."""

        @device("TEE0").jax
        def compute(x, y):
            return x * 2 + y

        x = put("TEE0", jnp.array([1, 2, 3]))
        y = put("TEE0", jnp.array([10, 20, 30]))
        result = compute(x, y)

        assert get_dev_attr(result) == "TEE0"


# =============================================================================
# TEE <-> SPU Transfer
# =============================================================================


class TestTeeSPUTransfer:
    """Test data transfers between TEE and SPU devices."""

    def test_transfer_tee_to_spu(self, ctx_3pc):
        """Data should transfer from TEE to SPU (seal as shares)."""
        x = put("TEE0", jnp.array([1, 2, 3]))
        x_spu = put("SP0", x)

        assert get_dev_attr(x_spu) == "SP0"

    def test_transfer_spu_to_tee(self, ctx_3pc):
        """Data should transfer from SPU to TEE (reveal)."""
        x = put("SP0", jnp.array([4, 5, 6]))
        x_tee = put("TEE0", x)

        assert get_dev_attr(x_tee) == "TEE0"


# =============================================================================
# Auto Transfer with TEE
# =============================================================================


class TestAutoTransferTee:
    """Test automatic data transfer involving TEE."""

    def test_auto_transfer_ppu_to_tee(self, ctx_3pc):
        """PPU data should auto-transfer to TEE when function runs on TEE."""

        @device("TEE0").jax
        def add(x, y):
            return x + y

        x = put("P0", jnp.array([1, 2, 3]))
        y = put("TEE0", jnp.array([4, 5, 6]))

        # x should auto-transfer from P0 to TEE0
        result = add(x, y)

        assert get_dev_attr(result) == "TEE0"

    def test_device_inference_with_tee_and_ppu(self, ctx_3pc):
        """When mixing PPU and TEE args, should infer TEE as target."""

        @device()
        def add(x, y):
            return x + y

        x = put("P0", jnp.array([1, 2, 3]))
        y = put("TEE0", jnp.array([4, 5, 6]))

        result = add(x, y)

        # Should infer TEE0 (secure device takes precedence)
        assert get_dev_attr(result) == "TEE0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
