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

import jax.numpy as jnp
import pytest

from mplang.frontend import phe


def test_mul_float_not_supported():
    """Test that multiplication with floats raises an error in frontend."""
    # Test with float ciphertext
    # TODO(jint): this assumption is wrong, Additive PHE can not handle trunction, so
    # it can not support flp x flp, but it can support flp x int
    float_ct = jnp.array(5.5, dtype=jnp.float32)
    int_pt = jnp.array(3, dtype=jnp.int32)

    with pytest.raises(
        ValueError,
        match="PHE multiplication does not support floating-point numbers",
    ):
        phe.mul(float_ct, int_pt)

    # Test with float plaintext
    int_ct = jnp.array(5, dtype=jnp.int32)
    float_pt = jnp.array(3.2, dtype=jnp.float32)

    with pytest.raises(
        ValueError,
        match="PHE multiplication does not support floating-point numbers",
    ):
        phe.mul(int_ct, float_pt)

    # Test with both floats
    with pytest.raises(
        ValueError,
        match="PHE multiplication does not support floating-point numbers",
    ):
        phe.mul(float_ct, float_pt)
