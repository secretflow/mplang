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

"""Verify PSI OKVS Logic Single Threaded.
Simulates psi_okvs.py logic flow without SimpSimulator.
"""

import os
import sys

# Ensure we can find the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import jax.numpy as jnp
import numpy as np

import mplang.v2.edsl.typing as elt
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import field, tensor
from mplang.v2.runtime.interpreter import InterpObject, Interpreter


def main():
    print("Beginning Single-Threaded PSI OKVS Verification...")

    interp = Interpreter()

    N = 100
    M = 140

    # 1. Inputs
    keys = np.random.randint(0, 1000000, size=(N,), dtype=np.uint64)

    # helper
    def to_obj(arr):
        return InterpObject(
            TensorValue(jnp.array(arr)), elt.Tensor[elt.u64, arr.shape], interp
        )

    key_obj = to_obj(keys)

    # 2. Simulate VOLE
    # Sender U, V. Recv Delta, W = V + U*Delta
    u_val = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(M, 2), dtype=np.uint64)
    v_val = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(M, 2), dtype=np.uint64)
    delta_val = np.array([5, 0], dtype=np.uint64)  # Delta=5

    u_obj = to_obj(u_val)
    v_obj = to_obj(v_val)
    delta_obj = to_obj(delta_val)

    with interp:
        # Receiver Computes W
        # Manually expand delta for logic check
        def _expand(d, ref):
            return jnp.tile(d, (ref.shape[0], 1))

        d_exp = tensor.run_jax(_expand, delta_obj, u_obj)
        prod = field.mul(u_obj, d_exp)
        w_obj = field.add(v_obj, prod)

        w_val_recv = w_obj.runtime_obj.unwrap()  # Recv gets this

    w_recv_obj = to_obj(w_val_recv)

    # 3. Receiver Logic (psi_okvs._recv_ops)
    # We call the exact function if possible, or replicate it.
    # psi_okvs._recv_ops is inner function.
    # We replicate logic to ensure our understanding matches implementation.

    print("Running Receiver Ops...")

    with interp:
        # Pre-process keys to get inputs for OKVS
        # We need keys and values (target).
        # Target = Hash(key).
        # Use aes_expand on keys directly (primitive).

        # Prepare inputs for aes_expand
        def _prep_seeds(items):
            items.shape[0]
            lo = items
            hi = jnp.zeros_like(items)
            # return (N, 2)
            return jnp.stack([lo, hi], axis=1)

        seeds_obj = tensor.run_jax(_prep_seeds, key_obj)

        # Call primitive
        h_y_expanded = field.aes_expand(seeds_obj, 1)

        # Convert (N, 1, 2) back to (N, 2)
        def _reshape_back(exp):
            return exp.reshape(exp.shape[0], 2)

        h_y = tensor.run_jax(_reshape_back, h_y_expanded)

        # OKVS Solve
        p_storage = field.solve_okvs(key_obj, h_y, m=M)

        # Mask
        q_storage = field.add(p_storage, w_recv_obj)

        q_val_sent = q_storage.runtime_obj.unwrap()

    q_sender_obj = to_obj(q_val_sent)

    # 4. Sender Logic
    print("Running Sender Ops...")
    with interp:
        # Decode Q
        s_decoded = field.decode_okvs(key_obj, q_sender_obj)

        # Decode V
        v_decoded = field.decode_okvs(key_obj, v_obj)

        # Hash items (Same as Recevier)
        seeds_obj_s = tensor.run_jax(_prep_seeds, key_obj)
        h_x_exp = field.aes_expand(seeds_obj_s, 1)
        h_x = tensor.run_jax(_reshape_back, h_x_exp)

        # T = S ^ V ^ H(x)
        t_val = field.add(s_decoded, v_decoded)
        t_val = field.add(t_val, h_x)

        u_decoded = field.decode_okvs(key_obj, u_obj)

        # Verify T == U_decoded * Delta
        final_t = t_val.runtime_obj.unwrap()
        u_decoded.runtime_obj.unwrap()

        tensor.run_jax(_expand, delta_obj, u_obj)  # Expand to N? No to M?

        # Expand Delta to N this time, as U_decoded is (N, 2)
        # Re-run expand with u_decoded shape (N, 2)
        def _expand_n(d, ref):
            return jnp.tile(d, (ref.shape[0], 1))

        d_exp_n = tensor.run_jax(
            _expand_n, delta_obj, key_obj
        )  # Use key_obj for N size

        ref_prod = field.mul(u_decoded, d_exp_n)
        ref_val = ref_prod.runtime_obj.unwrap()

        np.testing.assert_array_equal(
            final_t, ref_val, err_msg="PSI OKVS Verification Failed"
        )
        print("SUCCESS: PSI OKVS Logic Verified (T = U*Delta)")


if __name__ == "__main__":
    main()
