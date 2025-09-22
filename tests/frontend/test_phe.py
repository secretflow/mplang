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

"""Unit tests for PHE frontend."""

import pytest

from mplang.core.dtype import FLOAT32, INT32
from mplang.frontend import phe
from tests.frontend.dummy import DummyTensor


def test_mul_validation():
    """Test PHE multiplication validation logic."""

    # Test that float x float is blocked (requires truncation)
    # Under strict typed_op semantics positional args must be MPObject or TypeSpecs.
    # We validate purely at type level here using scalar TensorType placeholders.
    float_ct = DummyTensor(FLOAT32, ())
    float_pt = DummyTensor(FLOAT32, ())

    with pytest.raises(
        ValueError,
        match="PHE multiplication does not support float x float operations",
    ):
        phe.mul(float_ct, float_pt)

    # Test that float x int is allowed (no truncation required)
    int_pt = DummyTensor(INT32, ())

    # This should not raise a validation error (may fail for other reasons like missing keys)
    try:
        phe.mul(float_ct, int_pt)
    except ValueError as e:
        # Should not be the float x float validation error
        assert "float x float operations" not in str(e)
    except Exception:
        # Other exceptions are acceptable for this validation test
        pass

    # Test that int x float is allowed (no truncation required)
    int_ct = DummyTensor(INT32, ())

    try:
        phe.mul(int_ct, float_pt)
    except ValueError as e:
        # Should not be the float x float validation error
        assert "float x float operations" not in str(e)
    except Exception:
        # Other exceptions are acceptable for this validation test
        pass
