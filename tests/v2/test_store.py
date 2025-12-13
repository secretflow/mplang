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

import os
import shutil
import tempfile

import numpy as np
import pytest

from mplang.v2.dialects import tensor
from mplang.v2.dialects.store import load, save


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_save_load_tensor(simp_simulator_default, temp_dir):
    # 1. Create a tensor
    data = np.array([1, 2, 3], dtype=np.float32)
    x = tensor.constant(data)

    # 2. Save
    save_path = os.path.join(temp_dir, "my_tensor")
    save(x, save_path)

    # 3. Verify Shards exist (SimpSimulator runs locally)

    # 4. Load (with explicit type)
    y = load(save_path, expected_type=x.type)

    # 5. Verify content
    # y is likely a DriverVar in SimpSimulator containing URIs
    from mplang.v2.backends.simp_driver import DriverVar
    from mplang.v2.edsl import get_default_context

    sim = get_default_context()

    if isinstance(y, DriverVar):
        # Check party 0
        uri = y[0]

        # Resolve URI using the worker's store
        # We assume party 0 corresponds to worker 0
        if hasattr(sim, "context") and hasattr(sim.context, "workers"):
            worker = sim.context.workers[0]
            val = worker.store.get(uri)
            np.testing.assert_allclose(val.unwrap(), data)
        else:
            # If not SimpSimulator (e.g. some other context), try to unwrap directly
            np.testing.assert_allclose(y[0].unwrap(), data)

    else:
        # Fallback if it returns something else (e.g. MPObject wrapper?)
        if hasattr(y, "runtime_obj"):
            # If wrapped in InterpObject
            runtime_obj = y.runtime_obj
            if isinstance(runtime_obj, DriverVar):
                uri = runtime_obj[0]
                if hasattr(sim, "context") and hasattr(sim.context, "workers"):
                    worker = sim.context.workers[0]
                    val = worker.store.get(uri)
                    np.testing.assert_allclose(val.unwrap(), data)
                else:
                    np.testing.assert_allclose(runtime_obj[0].unwrap(), data)
            else:
                np.testing.assert_allclose(runtime_obj.unwrap(), data)
        else:
            # Maybe it's the value directly?
            # If it's a TensorValue
            if hasattr(y, "unwrap"):
                np.testing.assert_allclose(y.unwrap(), data)
            else:
                raise ValueError(f"Unknown return type from load: {type(y)}")
