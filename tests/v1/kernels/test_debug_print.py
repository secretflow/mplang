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


import numpy as np
import pytest

import mplang.v1 as mp
from mplang.v1.core import peval
from mplang.v1.ops import basic
from mplang.v1.runtime.simulation import Simulator


def _run_op(fe_op, *args, **kwargs):
    """Helper to run a frontend operation using peval."""
    pfunc, eval_args, out_tree = fe_op(*args, **kwargs)
    results = peval(pfunc, eval_args)
    return out_tree.unflatten(results)


def test_debug_print_tensor_pass_through():
    sim = Simulator.simple(3)
    mp.set_ctx(sim)
    x = _run_op(basic.prand, shape=(2,))
    # Should not raise and should return the same shaped/type object
    y = _run_op(basic.debug_print, x, prefix="x=")
    # Fetch to ensure the graph executed and values exist per party
    vals = mp.fetch(sim, y)
    assert isinstance(vals, list) and len(vals) == sim.world_size()
    # Some parties may not hold values depending on pmask; prand should set all enabled
    # After fetch, Values are normalized to numpy arrays
    for v in vals:
        if v is not None:
            # fetch returns numpy arrays via normalization
            assert isinstance(v, np.ndarray)
            assert v.shape == (2,)


def test_debug_print_table_pass_through():
    from mplang.v1.kernels.value import TableValue

    try:
        import pandas as pd
    except Exception:
        pytest.skip("pandas not available in test environment")
    sim = Simulator.simple(2)
    mp.set_ctx(sim)
    # Build a small DataFrame constant
    df = pd.DataFrame({"a": [1, 2, 3]})
    t = _run_op(basic.constant, df)
    t2 = _run_op(basic.debug_print, t, prefix="t=")
    vals = mp.fetch(sim, t2)
    # Both ranks hold the constant
    assert all(v is not None for v in vals)
    # Values are now returned as TableValue
    for v in vals:
        assert isinstance(v, TableValue)
        result_df = v.to_pandas()
        assert isinstance(result_df, pd.DataFrame)
