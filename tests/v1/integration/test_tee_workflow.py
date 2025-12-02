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

import mplang
import mplang.v1 as mp
from mplang.v1.ops import basic, crypto, tee


def _demo_flow():
    P0, P1, P2 = 0, 1, 2
    # TEE generates two ephemeral keypairs and quotes binding their pk
    t_sk0, t_pk0 = mp.run_at(P2, crypto.kem_keygen, "x25519")
    t_sk1, t_pk1 = mp.run_at(P2, crypto.kem_keygen, "x25519")
    q0 = mp.run_at(P2, tee.quote_gen, t_pk0)
    q1 = mp.run_at(P2, tee.quote_gen, t_pk1)

    # Send quotes to P0/P1 and attest to obtain TEE public keys
    q0_for_p0 = mp.p2p(P2, P0, q0)
    q1_for_p1 = mp.p2p(P2, P1, q1)
    t_pk0_for_p0 = mp.run_at(P0, tee.attest, q0_for_p0)
    t_pk1_for_p1 = mp.run_at(P1, tee.attest, q1_for_p1)

    # Each party generates its own ephemeral keypair and shares pk with TEE
    v_sk0, v_pk0 = mp.run_at(P0, crypto.kem_keygen, "x25519")
    v_sk1, v_pk1 = mp.run_at(P1, crypto.kem_keygen, "x25519")
    v_pk0_at_tee = mp.p2p(P0, P2, v_pk0)
    v_pk1_at_tee = mp.p2p(P1, P2, v_pk1)

    # Derive shared secrets on both sides and HKDF to session keys
    shared0_v = mp.run_at(P0, crypto.kem_derive, v_sk0, t_pk0_for_p0, "x25519")
    shared1_v = mp.run_at(P1, crypto.kem_derive, v_sk1, t_pk1_for_p1, "x25519")
    shared0_t = mp.run_at(P2, crypto.kem_derive, t_sk0, v_pk0_at_tee, "x25519")
    shared1_t = mp.run_at(P2, crypto.kem_derive, t_sk1, v_pk1_at_tee, "x25519")

    info_literal = "mplang/device/tee/v1"
    sess0_v = mp.run_at(P0, crypto.hkdf, shared0_v, info_literal)
    sess1_v = mp.run_at(P1, crypto.hkdf, shared1_v, info_literal)
    sess0_t = mp.run_at(P2, crypto.hkdf, shared0_t, info_literal)
    sess1_t = mp.run_at(P2, crypto.hkdf, shared1_t, info_literal)

    # Encrypt data on clients
    x0 = mp.run_jax_at(P0, lambda: np.array([10, 20, 30], dtype=np.uint8))
    x1 = mp.run_jax_at(P1, lambda: np.array([1, 2, 3], dtype=np.uint8))
    b0 = mp.run_at(P0, basic.pack, x0)
    b1 = mp.run_at(P1, basic.pack, x1)
    c0 = mp.run_at(P0, crypto.enc, b0, sess0_v)
    c1 = mp.run_at(P1, crypto.enc, b1, sess1_v)
    c0_at_tee = mp.p2p(P0, P2, c0)
    c1_at_tee = mp.p2p(P1, P2, c1)
    b0_at_tee = mp.run_at(P2, crypto.dec, c0_at_tee, sess0_t)
    b1_at_tee = mp.run_at(P2, crypto.dec, c1_at_tee, sess1_t)

    p0 = mp.run_at(P2, basic.unpack, b0_at_tee, out_ty=mp.TensorType.from_obj(x0))
    p1 = mp.run_at(P2, basic.unpack, b1_at_tee, out_ty=mp.TensorType.from_obj(x1))
    return p0, p1


def test_crypto_enc_dec_and_tee_quote_attest_roundtrip():
    # Create simulator with TEE bindings using the new initial_bindings parameter
    tee_bindings = {
        "tee.quote_gen": "mock_tee.quote_gen",
        "tee.attest": "mock_tee.attest",
    }
    sim = mplang.Simulator.simple(3, op_bindings=tee_bindings)
    p0, p1 = mplang.evaluate(sim, _demo_flow)
    a = mplang.fetch(sim, p0)
    b = mplang.fetch(sim, p1)
    # Expect third element to be the numpy arrays propagated
    assert np.array_equal(a[2], np.array([10, 20, 30], dtype=np.uint8))
    assert np.array_equal(b[2], np.array([1, 2, 3], dtype=np.uint8))
