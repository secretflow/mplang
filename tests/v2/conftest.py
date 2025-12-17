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

import warnings

import jax
import pytest

# Disable JAX persistent compilation cache to avoid warnings in tests
# (cache fails when jax_compilation_cache_dir is not configured)
jax.config.update("jax_enable_compilation_cache", False)

# Suppress known TEE mock warnings in tests (expected for local testing)
warnings.filterwarnings("ignore", message=".*Insecure mock TEE.*")

import mplang.v2.edsl.context  # noqa: E402
from mplang.v2.dialects import simp  # noqa: E402
from tests.v2.utils.tensor_patch import patch_object_operators  # noqa: E402

# Apply tensor operator overloading patch for tests
patch_object_operators()


@pytest.fixture
def simp_simulator_default():
    """Provide a SIMP simulator context for tests that need it.

    Usage: Add @pytest.mark.usefixtures("simp_simulator_default") to test classes
    or use simp_simulator_default fixture parameter in test functions.
    """
    sim = simp.make_simulator(world_size=3)
    with sim:
        yield sim


@pytest.fixture(autouse=True)
def reset_context_stack():
    """Reset the context stack before and after each test."""
    # Clear context stack for test isolation
    mplang.v2.edsl.context._context_stack.clear()
    yield
    # Clear again after test
    mplang.v2.edsl.context._context_stack.clear()
