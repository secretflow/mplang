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

# Integration tests for mock crypto pipeline (frontend -> runtime -> backend)
# Marked as integration due to use of Simulator and multi-party coordination.

from __future__ import annotations

import numpy as np
import pytest

import mplang.v1 as mp
from mplang.v1.ops import basic, crypto  # Frontend modules not in __init__

pytestmark = pytest.mark.integration


def _mk_sim(n: int = 3):
    return mp.Simulator.simple(n)


def test_keygen_shape():
    sim = _mk_sim()

    @mp.function
    def fn():
        k16 = mp.run_at(0, crypto.keygen, 16)
        k32 = mp.run_at(1, crypto.keygen)
        return k16, k32

    k16, k32 = mp.evaluate(sim, fn)
    a16, a32 = mp.fetch(sim, (k16, k32))
    assert a16[0].shape == (16,)
    assert a32[1].shape == (32,)
    assert a16[0].dtype == np.uint8


def test_enc_dec_roundtrip_bytes():
    sim = _mk_sim()

    @mp.function
    def fn():
        key = mp.run_at(0, crypto.keygen, 32)
        pt = mp.run_jax_at(0, lambda: np.arange(50, dtype=np.uint8))
        ct = mp.run_at(0, crypto.enc, pt, key)
        rt = mp.run_at(0, crypto.dec, ct, key)
        return pt, rt

    pt, rt = mp.evaluate(sim, fn)
    pt_v, rt_v = mp.fetch(sim, (pt, rt))
    np.testing.assert_array_equal(pt_v[0], rt_v[0])


def test_pack_unpack_roundtrip_various():
    sim = _mk_sim()

    shapes_dtypes = [
        ((), np.int32),
        ((4,), np.float32),
        ((2, 3), np.int64),
    ]

    @mp.function
    def fn():
        outs = []
        for _idx, (shape, dt) in enumerate(shapes_dtypes):
            arr = mp.run_jax_at(0, lambda s=shape, d=dt: np.zeros(s, dtype=d))
            packed = mp.run_at(0, basic.pack, arr)
            unpacked = mp.run_at(
                0, basic.unpack, packed, out_ty=mp.TensorType.from_obj(arr)
            )
            outs.append(unpacked)
        return tuple(outs)

    (u0, u1, u2) = mp.evaluate(sim, fn)
    (v0, v1, v2) = mp.fetch(sim, (u0, u1, u2))
    assert v0[0].shape == ()
    assert v1[0].shape == (4,)
    assert v2[0].shape == (2, 3)


def test_kem_hkdf_symmetric_mock():
    sim = _mk_sim()

    @mp.function
    def fn():
        sk0, pk0 = mp.run_at(0, crypto.kem_keygen)
        sk1, pk1 = mp.run_at(1, crypto.kem_keygen)
        pk1_on_0 = mp.p2p(1, 0, pk1)
        pk0_on_1 = mp.p2p(0, 1, pk0)
        sec0 = mp.run_at(0, crypto.kem_derive, sk0, pk1_on_0)
        sec1 = mp.run_at(1, crypto.kem_derive, sk1, pk0_on_1)
        k0 = mp.run_at(0, crypto.hkdf, sec0, "info")
        k1 = mp.run_at(1, crypto.hkdf, sec1, "info")
        return k0, k1

    k0, k1 = mp.evaluate(sim, fn)
    v0, v1 = mp.fetch(sim, (k0, k1))
    np.testing.assert_array_equal(v0[0], v1[1])


@pytest.mark.parametrize("length", [1, 7, 32, 64])
def test_variable_key_lengths(length: int):
    sim = _mk_sim()

    @mp.function
    def fn(n: int):
        return mp.run_at(0, crypto.keygen, n)

    k = mp.evaluate(sim, fn, length)
    v = mp.fetch(sim, k)
    assert v[0].shape == (length,)
    assert v[0].dtype == np.uint8
