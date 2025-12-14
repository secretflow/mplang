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

"""VOLE correctness test using mp API."""

import numpy as np

import mplang.v2 as mp
from mplang.v2.dialects import simp, tensor
from mplang.v2.libs.mpc.vole import gilboa as vole


def test_vole_correctness():
    """Verify VOLE correlation: w = v + u * delta."""
    sim = simp.make_simulator(2)
    mp.set_root_context(sim)
    N = 100

    def protocol():
        # Providers
        def u_provider():
            u_val = np.ones((N, 2), dtype=np.uint64)
            u_val[:, 1] = 0
            return tensor.constant(u_val)  # On Sender

        def delta_provider():
            delta_val = np.array([3, 0], dtype=np.uint64)
            return tensor.constant(delta_val)  # On Receiver

        # Run VOLE (Sender=0, Receiver=1)
        v_sender, w_recv = vole.vole(0, 1, N, u_provider, delta_provider)

        # We need to return them to check
        # v is on 0. w is on 1.
        return v_sender, w_recv

    traced = mp.compile(protocol)
    v_obj, w_obj = mp.evaluate(traced)

    # Fetch results
    v = mp.fetch(v_obj)[0]  # v on party 0
    w = mp.fetch(w_obj)[1]  # w on party 1

    # Reconstruct inputs for check
    u = np.ones((N, 2), dtype=np.uint64)
    u[:, 1] = 0
    delta = np.array([3, 0], dtype=np.uint64)

    print(f"v shape: {v.shape}")
    print(f"w shape: {w.shape}")

    # Verify: w = v + u * delta
    # Need to do GF128 arithmetic on host to verify
    from mplang.v2.backends.field_impl import _gf128_mul_impl

    # Get bit decomposition of delta for debugging
    delta_bits = np.unpackbits(delta.view(np.uint8), bitorder="little")

    mismatch_count = 0
    for i in range(N):
        prod = _gf128_mul_impl(u[i], delta)
        term = v[i] ^ prod
        if not np.array_equal(term, w[i]):
            d_i = delta_bits[i] if i < 128 else 0
            print(f"Mismatch at {i} (d={d_i}): w={w[i]}, v={v[i]}")
            mismatch_count += 1

    if mismatch_count > 0:
        print(f"Total Mismatches: {mismatch_count}")
        raise AssertionError

    print("VOLE Verification PASSED.")


if __name__ == "__main__":
    test_vole_correctness()
