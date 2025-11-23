"""Tests for Homomorphic Aggregation library."""

import numpy as np

import mplang2.backends.bfv_impl
import mplang2.backends.tensor_impl  # noqa: F401
from mplang2.backends.simp_simulator import SimpSimulator, get_or_create_context
from mplang2.dialects import bfv, tensor
from mplang2.libs import aggregation


def test_rotate_and_sum():
    # World size 1 (Local BFV simulation)
    SimpSimulator(world_size=1)
    get_or_create_context(1)

    # Setup BFV
    # We need to run this inside the interpreter context or just use the primitives
    # if the backend is registered.
    # Since we are using SimpSimulator, we can run BFV ops.

    # 1. Keygen
    pk, sk = bfv.keygen(poly_modulus_degree=4096)
    gk = bfv.make_galois_keys(sk)
    encoder = bfv.create_encoder(poly_modulus_degree=4096)

    # 2. Data: [1, 2, 3, 4, 0, ...]
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    # Pad to 4096? No, encode handles it.

    t_data = tensor.constant(data)
    pt = bfv.encode(t_data, encoder)
    ct = bfv.encrypt(pt, pk)

    # 3. Aggregate first 4 elements
    # Expected sum: 1+2+3+4 = 10

    def protocol(ct, gk):
        return aggregation.rotate_and_sum(ct, k=4, galois_keys=gk)

    # We need to wrap these in InterpObjects if we run via interpreter,
    # or just call them if we are in eager mode with direct execution?
    # The BFV backend implementation usually returns opaque objects (wrappers).
    # Let's see how `test_bfv.py` does it.
    # It usually calls `bfv.add(ct1, ct2)` directly.

    # However, `rotate_and_sum` uses `bfv.rotate` and `bfv.add`.
    # If we call `protocol(ct, gk)`, it will execute immediately.

    res_ct = protocol(ct, gk)

    # Decrypt and check
    res_pt = bfv.decrypt(res_ct, sk)
    res_tensor = bfv.decode(res_pt, encoder)

    # res_tensor is an InterpObject.
    # In local simulation, runtime_obj should be the numpy array.
    assert res_tensor.runtime_obj[0] == 10


def test_masked_aggregate():
    SimpSimulator(world_size=1)
    get_or_create_context(1)

    pk, sk = bfv.keygen(poly_modulus_degree=4096)
    encoder = bfv.create_encoder(poly_modulus_degree=4096)

    # CT1: [10, 0, 0, ...] -> Mask1: [1, 0, 0, ...]
    # CT2: [0, 20, 0, ...] -> Mask2: [0, 1, 0, ...]

    data1 = np.array([10, 0, 0, 0], dtype=np.int64)
    data2 = np.array([0, 20, 0, 0], dtype=np.int64)

    mask1 = np.array([1, 0, 0, 0], dtype=np.int64)
    mask2 = np.array([0, 1, 0, 0], dtype=np.int64)

    ct1 = bfv.encrypt(bfv.encode(tensor.constant(data1), encoder), pk)
    ct2 = bfv.encrypt(bfv.encode(tensor.constant(data2), encoder), pk)

    pt_mask1 = bfv.encode(tensor.constant(mask1), encoder)
    pt_mask2 = bfv.encode(tensor.constant(mask2), encoder)

    def protocol(cts, masks):
        return aggregation.masked_aggregate(cts, masks)

    res_ct = protocol([ct1, ct2], [pt_mask1, pt_mask2])

    res_pt = bfv.decrypt(res_ct, sk)
    res = bfv.decode(res_pt, encoder)

    assert res.runtime_obj[0] == 10
    assert res.runtime_obj[1] == 20
