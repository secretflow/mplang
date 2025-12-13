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

import pytest

import mplang.v2.edsl.context
from tests.v2.utils.tensor_patch import patch_object_operators


@pytest.fixture
def simp_simulator_default(monkeypatch):
    """Temporarily register SimpSimulator as the default interpreter."""
    from mplang.v2.dialects import simp

    monkeypatch.setattr(
        mplang.v2.edsl.context,
        "_default_context_factory",
        lambda: simp.make_simulator(world_size=3),
    )
    monkeypatch.setattr(mplang.v2.edsl.context, "_default_context", None)


# Apply tensor operator overloading patch for tests
patch_object_operators()


@pytest.fixture(autouse=True)
def reset_default_context(monkeypatch):
    """Reset the default context and context stack before and after each test."""
    # Clear context stack for test isolation
    mplang.v2.edsl.context._context_stack.clear()
    monkeypatch.setattr(mplang.v2.edsl.context, "_default_context", None)
    yield
    # Clear again after test
    mplang.v2.edsl.context._context_stack.clear()
