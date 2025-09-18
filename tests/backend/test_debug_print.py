# Copyright 2025 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0

import numpy as np
import pytest

import mplang as mp
from mplang.core import constant, debug_print, prand
from mplang.runtime.simulation import Simulator


def test_debug_print_tensor_pass_through():
    sim = Simulator.simple(3)
    mp.set_ctx(sim)
    x = prand((2,))
    # Should not raise and should return the same shaped/type object
    y = debug_print(x, prefix="x=")
    # Fetch to ensure the graph executed and values exist per party
    vals = mp.fetch(sim, y)
    assert isinstance(vals, list) and len(vals) == sim.world_size()
    # Some parties may not hold values depending on pmask; prand should set all enabled
    # So we allow None only if pmask excludes ranks; otherwise expect numpy arrays
    for v in vals:
        if v is not None:
            assert isinstance(v, np.ndarray)
            assert v.shape == (2,)


def test_debug_print_table_pass_through():
    try:
        import pandas as pd
    except Exception:
        pytest.skip("pandas not available in test environment")
    sim = Simulator.simple(2)
    mp.set_ctx(sim)
    # Build a small DataFrame constant
    df = pd.DataFrame({"a": [1, 2, 3]})
    t = constant(df)
    t2 = debug_print(t, prefix="t=")
    vals = mp.fetch(sim, t2)
    # Both ranks hold the constant
    assert all(v is not None for v in vals)
    # Verify type is DataFrame
    for v in vals:
        assert isinstance(v, pd.DataFrame)
