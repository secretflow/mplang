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

    # 1) TEE party generates two ephemeral keypairs (KEM-style)
    # In real impl, quote will bind H(t_pk) in report_data.
    t_sk0, t_pk0 = simp.runAt(P2, crypto.kem_keygen)("x25519")
    t_sk1, t_pk1 = simp.runAt(P2, crypto.kem_keygen)("x25519")

    # 2) TEE generates quotes that carry (or bind) the public materials
    quotes = simp.runAt(P2, tee.quote)([t_pk0, t_pk1])

    # Scatter quotes to data providers
    q_for_p0 = simp.p2p(P2, P0, quotes[0])
    q_for_p1 = simp.p2p(P2, P1, quotes[1])

    # 3) Data providers verify their quotes (gating). In a real impl, attest
    # would return the attested TEE public key. Our mock returns a tiny tensor,
    # so we perform an explicit p2p of the public material after this step.
    _ = simp.runAt(P0, tee.attest)(q_for_p0)
    _ = simp.runAt(P1, tee.attest)(q_for_p1)
    t_pk0_for_p0 = simp.p2p(P2, P0, t_pk0)
    t_pk1_for_p1 = simp.p2p(P2, P1, t_pk1)

    # 4) Each data provider generates its own ephemeral keypair
    v_sk0, v_pk0 = simp.runAt(P0, crypto.kem_keygen)("x25519")
    v_sk1, v_pk1 = simp.runAt(P1, crypto.kem_keygen)("x25519")

    # Send V-side public materials to TEE
    v_pk0_at_tee = simp.p2p(P0, P2, v_pk0)
    v_pk1_at_tee = simp.p2p(P1, P2, v_pk1)

    # 5) Derive per-party shared secrets, then HKDF to final session keys
    shared0_v = simp.runAt(P0, crypto.kem_derive)(v_sk0, t_pk0_for_p0, "x25519")
    shared1_v = simp.runAt(P1, crypto.kem_derive)(v_sk1, t_pk1_for_p1, "x25519")
    shared0_t = simp.runAt(P2, crypto.kem_derive)(t_sk0, v_pk0_at_tee, "x25519")
    shared1_t = simp.runAt(P2, crypto.kem_derive)(t_sk1, v_pk1_at_tee, "x25519")

    info_p0 = simp.runAt(P0, lambda: np.frombuffer(b"V->TEE", dtype=np.uint8))()
    info_p1 = simp.runAt(P1, lambda: np.frombuffer(b"V->TEE", dtype=np.uint8))()
    info_t = simp.runAt(P2, lambda: np.frombuffer(b"V->TEE", dtype=np.uint8))()

    sess0_v = simp.runAt(P0, crypto.hkdf)(shared0_v, info_p0)
    sess1_v = simp.runAt(P1, crypto.hkdf)(shared1_v, info_p1)
    sess0_t = simp.runAt(P2, crypto.hkdf)(shared0_t, info_t)
    sess1_t = simp.runAt(P2, crypto.hkdf)(shared1_t, info_t)

    # 6) Encrypt local data using derived session keys
    x0 = simp.runAt(P0, lambda: np.array([10, 20, 30], dtype=np.uint8))()
    x1 = simp.runAt(P1, lambda: np.array([1, 2, 3], dtype=np.uint8))()
    c0 = simp.runAt(P0, crypto.enc)(x0, sess0_v)
    c1 = simp.runAt(P1, crypto.enc)(x1, sess1_v)

    # 7) Send ciphertexts to TEE and decrypt using the same session keys at TEE
    c0_at_tee = simp.p2p(P0, P2, c0)
    c1_at_tee = simp.p2p(P1, P2, c1)

    p0 = simp.runAt(P2, crypto.dec)(c0_at_tee, sess0_t)
    p1 = simp.runAt(P2, crypto.dec)(c1_at_tee, sess1_t)

    # Return plaintexts reconstructed at TEE for quick visual check
    return p0, p1


if __name__ == "__main__":
    # Build a simple 3-party simulator
    sim = mplang.Simulator.simple(3)
    p0, p1 = mplang.evaluate(sim, demo_flow)
    print("tee_attestation (mock) results:")
    print("P0 plaintext:", mplang.fetch(sim, p0))
    print("P1 plaintext:", mplang.fetch(sim, p1))
