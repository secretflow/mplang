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
Minimal demonstration of TEE quote generation and verification (mock).

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

    # 1) TEE party generates two keys
    key_a = simp.runAt(P2, crypto.keygen)(16)
    key_b = simp.runAt(P2, crypto.keygen)(16)

    # 2) TEE generates quotes with keys as payloads
    quotes = simp.runAt(P2, tee.quote)([key_a, key_b])

    # Scatter quotes to data providers
    q_for_p0 = simp.p2p(P2, P0, quotes[0])
    q_for_p1 = simp.p2p(P2, P1, quotes[1])

    # 3) Data providers verify their quotes to obtain keys
    k0 = simp.runAt(P0, tee.attest)(q_for_p0)
    k1 = simp.runAt(P1, tee.attest)(q_for_p1)

    # 4) Encrypt local data using obtained keys
    x0 = simp.runAt(P0, lambda: np.array([10, 20, 30], dtype=np.uint8))()
    x1 = simp.runAt(P1, lambda: np.array([1, 2, 3], dtype=np.uint8))()
    c0 = simp.runAt(P0, crypto.enc)(x0, k0)
    c1 = simp.runAt(P1, crypto.enc)(x1, k1)

    # 5) Send ciphertexts to TEE and decrypt using original keys at TEE
    c0_at_tee = simp.p2p(P0, P2, c0)
    c1_at_tee = simp.p2p(P1, P2, c1)

    p0 = simp.runAt(P2, crypto.dec)(c0_at_tee, key_a)
    p1 = simp.runAt(P2, crypto.dec)(c1_at_tee, key_b)

    # Return plaintexts reconstructed at TEE for quick visual check
    return p0, p1


if __name__ == "__main__":
    # Build a simple 3-party simulator
    sim = mplang.Simulator.simple(3)
    p0, p1 = mplang.evaluate(sim, demo_flow)
    print("tee_attestation (mock) results:")
    print("P0 plaintext:", mplang.fetch(sim, p0))
    print("P1 plaintext:", mplang.fetch(sim, p1))
