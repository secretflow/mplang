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

"""Tests for device API error handling and edge cases.

This module tests:
- Custom exception class hierarchy
- Error conditions for put(), device(), and related functions
- Edge cases and boundary conditions
- Warning behaviors
"""

import warnings

import jax.numpy as jnp
import pytest

from mplang.v2.libs.device import (
    DeviceError,
    DeviceInferenceError,
    DeviceNotFoundError,
    FrontendError,
    device,
    get_dev_attr,
    is_device_obj,
    put,
    set_dev_attr,
)

# =============================================================================
# Exception Class Hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Test custom exception class inheritance and behavior."""

    def test_all_errors_inherit_from_device_error(self):
        """All device exceptions should inherit from DeviceError."""
        assert issubclass(DeviceNotFoundError, DeviceError)
        assert issubclass(DeviceInferenceError, DeviceError)
        assert issubclass(FrontendError, DeviceError)

    def test_device_error_is_exception(self):
        """DeviceError should be a standard Exception."""
        assert issubclass(DeviceError, Exception)

    @pytest.mark.parametrize(
        "exc_class,message",
        [
            (DeviceError, "base device error"),
            (DeviceNotFoundError, "device X not found"),
            (DeviceInferenceError, "ambiguous device"),
            (FrontendError, "invalid frontend"),
        ],
    )
    def test_exception_preserves_message(self, exc_class, message):
        """Exceptions should preserve their error messages."""
        exc = exc_class(message)
        assert message in str(exc)


# =============================================================================
# DeviceNotFoundError
# =============================================================================


class TestDeviceNotFoundError:
    """Test DeviceNotFoundError for invalid device IDs."""

    @pytest.mark.parametrize(
        "invalid_id",
        ["INVALID", "NONEXISTENT", "P99", "SPU_WRONG", ""],
    )
    def test_put_invalid_device_id(self, ctx_3pc, invalid_id):
        """put() should raise DeviceNotFoundError for invalid device IDs."""
        with pytest.raises(DeviceNotFoundError) as exc_info:
            put(invalid_id, jnp.array([1, 2, 3]))

        # Error message should mention the invalid ID and available devices
        error_msg = str(exc_info.value)
        if invalid_id:  # Skip check for empty string
            assert invalid_id in error_msg
        assert "Available devices" in error_msg

    def test_device_decorator_invalid_device(self, ctx_3pc):
        """device() decorator should raise DeviceNotFoundError at call time."""

        @device("NONEXISTENT")
        def fn(a, b):
            return a + b

        with pytest.raises(DeviceNotFoundError) as exc_info:
            fn(jnp.array([1]), jnp.array([2]))

        assert "NONEXISTENT" in str(exc_info.value)


# =============================================================================
# DeviceInferenceError
# =============================================================================


class TestDeviceInferenceError:
    """Test DeviceInferenceError for ambiguous device inference."""

    def test_no_device_objects_passed(self, ctx_3pc):
        """Should error when no device-bound objects are passed."""

        @device()
        def fn(a, b):
            return a + b

        with pytest.raises(DeviceInferenceError) as exc_info:
            fn(jnp.array([1]), jnp.array([2]))

        assert "no device-bound Object" in str(exc_info.value)

    def test_multiple_ppu_devices(self, ctx_3pc):
        """Should error when args come from multiple PPU devices."""

        @device()
        def add(a, b):
            return a + b

        x = put("P0", jnp.array([1]))
        y = put("P1", jnp.array([2]))

        with pytest.raises(DeviceInferenceError) as exc_info:
            add(x, y)

        assert "multiple PPU devices" in str(exc_info.value)

    def test_multiple_spu_devices(self, ctx_4pc):
        """Should error when args come from multiple SPU devices."""

        @device()
        def add(a, b):
            return a + b

        x = put("SP0", jnp.array([1.0]))
        y = put("SP1", jnp.array([2.0]))

        with pytest.raises(DeviceInferenceError) as exc_info:
            add(x, y)

        assert "multiple SPU devices" in str(exc_info.value)


# =============================================================================
# FrontendError
# =============================================================================


class TestFrontendError:
    """Test FrontendError for invalid frontend values."""

    @pytest.mark.parametrize(
        "invalid_frontend",
        ["unknown", "tensorflow", "pytorch", "numpy", "invalid_frontend"],
    )
    def test_unknown_frontend_raises_error(self, ctx_3pc, invalid_frontend):
        """Unknown frontend should raise FrontendError."""
        with pytest.raises(FrontendError) as exc_info:

            @device("P0", invalid_frontend)
            def fn(x):
                return x

        error_msg = str(exc_info.value)
        assert invalid_frontend in error_msg
        assert "Supported frontends" in error_msg

    def test_valid_frontend_jax(self, ctx_3pc):
        """Frontend 'jax' should work without errors."""

        @device("P0", "jax")
        def add(a, b):
            return a + b

        x = put("P0", jnp.array([1.0, 2.0]))
        y = put("P0", jnp.array([3.0, 4.0]))
        result = add(x, y)

        assert get_dev_attr(result) == "P0"


# =============================================================================
# Warnings
# =============================================================================


class TestWarnings:
    """Test warning behaviors in device API."""

    def test_spu_with_jax_frontend_warns(self, ctx_3pc):
        """Using frontend='jax' with SPU should emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @device("SP0", "jax")
            def add(a, b):
                return a + b

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            warning_msg = str(w[0].message).lower()
            assert "redundant" in warning_msg
            assert "sp0" in warning_msg


# =============================================================================
# Exception Catching Patterns
# =============================================================================


class TestExceptionCatching:
    """Test that exceptions can be caught at different levels."""

    @pytest.mark.parametrize(
        "trigger,expected_type",
        [
            ("invalid_device", DeviceNotFoundError),
            ("no_device_args", DeviceInferenceError),
            ("bad_frontend", FrontendError),
        ],
    )
    def test_specific_exceptions_raised(self, ctx_3pc, trigger, expected_type):
        """Each error condition should raise its specific exception type."""
        with pytest.raises(expected_type):
            if trigger == "invalid_device":
                put("NOPE", jnp.array([1]))
            elif trigger == "no_device_args":

                @device()
                def fn(a):
                    return a

                fn(jnp.array([1]))
            elif trigger == "bad_frontend":

                @device("P0", "bad")
                def fn2(a):
                    return a

    @pytest.mark.parametrize(
        "trigger",
        ["invalid_device", "no_device_args", "bad_frontend"],
    )
    def test_base_exception_catches_all(self, ctx_3pc, trigger):
        """DeviceError should catch all device-related exceptions."""
        with pytest.raises(DeviceError):
            if trigger == "invalid_device":
                put("NOPE", jnp.array([1]))
            elif trigger == "no_device_args":

                @device()
                def fn(a):
                    return a

                fn(jnp.array([1]))
            elif trigger == "bad_frontend":

                @device("P0", "bad")
                def fn2(a):
                    return a


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_device_none_none_with_inference(self, ctx_3pc):
        """device(None, None) should work with device inference."""
        x = put("P0", jnp.array([1]))

        @device(None, None)
        def identity(a):
            return a

        result = identity(x)
        assert get_dev_attr(result) == "P0"

    def test_explicit_device_overrides_arg_device(self, ctx_3pc):
        """Explicit device should override device inferred from args."""

        @device("P1")
        def move_to_p1(a):
            return a

        x = put("P0", jnp.array([1, 2, 3]))
        result = move_to_p1(x)

        # Result on P1 despite input being on P0
        assert get_dev_attr(result) == "P1"

    def test_device_context_default_values(self):
        """DeviceContext should have correct default values."""
        ctx1 = device("P0")
        ctx2 = device("P0", None)

        assert ctx1.dev_id == "P0"
        assert ctx1.frontend is None
        assert ctx2.dev_id == "P0"
        assert ctx2.frontend is None

    def test_device_context_with_frontend(self):
        """DeviceContext should store frontend correctly."""
        ctx = device("P0", "jax")
        assert ctx.dev_id == "P0"
        assert ctx.frontend == "jax"


# =============================================================================
# Helper Function Edge Cases
# =============================================================================


class TestHelperFunctions:
    """Test is_device_obj, set_dev_attr, get_dev_attr edge cases."""

    def test_is_device_obj_with_non_object(self):
        """is_device_obj should return False for non-Object types."""
        assert is_device_obj(None) is False
        assert is_device_obj(42) is False
        assert is_device_obj("string") is False
        assert is_device_obj([1, 2, 3]) is False
        assert is_device_obj(jnp.array([1])) is False

    def test_set_dev_attr_requires_object(self, ctx_3pc):
        """set_dev_attr should raise TypeError for non-Object."""
        with pytest.raises(TypeError, match="must be an instance of Object"):
            set_dev_attr(jnp.array([1]), "P0")

        with pytest.raises(TypeError):
            set_dev_attr("not an object", "P0")

    def test_get_dev_attr_requires_object(self):
        """get_dev_attr should raise TypeError for non-Object."""
        with pytest.raises(TypeError, match="must be an instance of Object"):
            get_dev_attr(jnp.array([1]))

    def test_get_dev_attr_requires_device_attribute(self, ctx_3pc):
        """get_dev_attr should raise ValueError if no device attribute."""
        # Put an object but then remove its device attribute
        x = put("P0", jnp.array([1, 2, 3]))
        delattr(x, "__device__")  # Remove the device attribute

        with pytest.raises(ValueError, match="does not have a device attribute"):
            get_dev_attr(x)

    def test_device_obj_lifecycle(self, ctx_3pc):
        """Test full lifecycle: put -> is_device_obj -> get_dev_attr."""
        x = put("P0", jnp.array([1, 2, 3]))

        assert is_device_obj(x) is True
        assert get_dev_attr(x) == "P0"


# =============================================================================
# TEE and Unimplemented Features
# =============================================================================


class TestUnimplementedFeatures:
    """Test that unimplemented features raise appropriate errors."""

    def test_tee_to_tee_transfer_not_implemented(self, ctx_multi_tee):
        """TEE to TEE device transfer should raise NotImplementedError."""
        x = put("TEE0", jnp.array([1, 2, 3]))

        with pytest.raises(NotImplementedError):
            put("TEE1", x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
