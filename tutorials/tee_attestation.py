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

"""
Minimal demonstration of TEE quote generation and verification with a mock
ECDH-style session derivation (no real ECDH: we derive a session key from
exchanged public materials using a hash for demo purposes).

Parties: P0, P1 are data providers; P2 plays the TEE party.
This tutorial uses the Simulator and the mock TeeHandler to emulate
quote generation and verification.
"""

from __future__ import annotations

import numpy as np

import mplang
import mplang.simp as simp
from mplang.frontend import crypto, tee


@mplang.function
def demo_flow():
    P0, P1, P2 = 0, 1, 2

    # 1) TEE generates ephemeral keypairs and quotes binding their pk
    tee_sk0, tee_pk0 = simp.runAt(P2, crypto.kem_keygen)("x25519")
    tee_sk1, tee_pk1 = simp.runAt(P2, crypto.kem_keygen)("x25519")
    quote_p0 = simp.runAt(P2, tee.quote)(tee_pk0)
    quote_p1 = simp.runAt(P2, tee.quote)(tee_pk1)

    # Scatter quotes to P0 & P1
    quote_p0_at_p0 = simp.p2p(P2, P0, quote_p0)
    quote_p1_at_p1 = simp.p2p(P2, P1, quote_p1)

    # 3) Data providers verify quotes and obtain the attested TEE public key
    tee_pk0_at_p0 = simp.runAt(P0, tee.attest)(quote_p0_at_p0)
    tee_pk1_at_p1 = simp.runAt(P1, tee.attest)(quote_p1_at_p1)

    # 4) Each data provider generates its own ephemeral keypair
    v_sk_p0, v_pk_p0 = simp.runAt(P0, crypto.kem_keygen)("x25519")
    v_sk_p1, v_pk_p1 = simp.runAt(P1, crypto.kem_keygen)("x25519")

    # Send V-side public materials to TEE
    v_pk_p0_at_tee = simp.p2p(P0, P2, v_pk_p0)
    v_pk_p1_at_tee = simp.p2p(P1, P2, v_pk_p1)

    # 5) Derive per-party shared secrets, then HKDF to final session keys
    shared0_p0 = simp.runAt(P0, crypto.kem_derive)(v_sk_p0, tee_pk0_at_p0, "x25519")
    shared1_p1 = simp.runAt(P1, crypto.kem_derive)(v_sk_p1, tee_pk1_at_p1, "x25519")
    shared0_tee = simp.runAt(P2, crypto.kem_derive)(tee_sk0, v_pk_p0_at_tee, "x25519")
    shared1_tee = simp.runAt(P2, crypto.kem_derive)(tee_sk1, v_pk_p1_at_tee, "x25519")

    info_p0 = simp.runAt(P0, lambda: np.frombuffer(b"V->TEE", dtype=np.uint8))()
    info_p1 = simp.runAt(P1, lambda: np.frombuffer(b"V->TEE", dtype=np.uint8))()
    info_tee = simp.runAt(P2, lambda: np.frombuffer(b"V->TEE", dtype=np.uint8))()

    sess0_p0 = simp.runAt(P0, crypto.hkdf)(shared0_p0, info_p0)
    sess1_p1 = simp.runAt(P1, crypto.hkdf)(shared1_p1, info_p1)
    sess0_tee = simp.runAt(P2, crypto.hkdf)(shared0_tee, info_tee)
    sess1_tee = simp.runAt(P2, crypto.hkdf)(shared1_tee, info_tee)

    # 6) Encrypt local data using derived session keys
    x_p0 = simp.runAt(P0, lambda: np.array([10, 20, 30], dtype=np.uint8))()
    x_p1 = simp.runAt(P1, lambda: np.array([1, 2, 3], dtype=np.uint8))()
    ct_p0 = simp.runAt(P0, crypto.enc)(x_p0, sess0_p0)
    ct_p1 = simp.runAt(P1, crypto.enc)(x_p1, sess1_p1)

    # 7) Send ciphertexts to TEE and decrypt using the same session keys at TEE
    ct_p0_at_tee = simp.p2p(P0, P2, ct_p0)
    ct_p1_at_tee = simp.p2p(P1, P2, ct_p1)

    pt0_at_tee = simp.runAt(P2, crypto.dec)(ct_p0_at_tee, sess0_tee)
    pt1_at_tee = simp.runAt(P2, crypto.dec)(ct_p1_at_tee, sess1_tee)

    # Return plaintexts reconstructed at TEE for quick visual check
    return pt0_at_tee, pt1_at_tee


if __name__ == "__main__":
    # Build a simple 3-party simulator
    sim = mplang.Simulator.simple(3)
    pt0_at_tee, pt1_at_tee = mplang.evaluate(sim, demo_flow)
    print("tee_attestation (mock) results:")
    print("P0 plaintext:", mplang.fetch(sim, pt0_at_tee))
    print("P1 plaintext:", mplang.fetch(sim, pt1_at_tee))
