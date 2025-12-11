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

import jax.numpy as jnp
import numpy as np

import mplang.v2.libs.mpc.vole.gilboa as vole
from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.runtime.interpreter import Interpreter


def prove_logic():
    print("Beginning Single-Threaded Logic Verification...")

    # 1. Setup Parameters
    N = 100

    # 2. Generate OT correlation (IKNP style)
    # Delta (Receiver choice)
    delta_val = np.array([3, 0], dtype=np.uint64)  # Delta = 3
    # Decompose delta to bits
    delta_bits_u8 = np.unpackbits(delta_val.view(np.uint8), bitorder="little")  # (128,)
    d_bits = delta_bits_u8.reshape(128, 1)  # (128, 1)

    # Base OT Choices (Sender in VOLE = Receiver in Base OT? No.)
    # In VOLE:
    # Receiver holds Delta. Receiver plays Receiver in OT extension.
    # Sender plays Sender in OT extension.
    # Standard IKNP:
    #   Sender has T (matrix), S (base choice).
    #   Receiver has Q. Q = T ^ (choice * S)
    #   Here choice IS delta bits.

    # Sender Secrets
    # T: (128, 128) bit matrix.
    # S: (128,) bit vector (Base OT choices).
    np.random.seed(42)
    t_matrix = np.random.randint(0, 2, size=(128, 128), dtype=np.uint8)
    s_base = np.random.randint(0, 2, size=(128,), dtype=np.uint8)

    # Receiver Secrets
    # Q = T ^ (d_bits * S)
    # d_bits is (128, 1). S is (128,). Broadcast logic:
    # Q_i = T_i ^ (d_i * S)
    # d_i * S is Scalar * Vector -> Vector (128,)

    q_matrix = np.zeros_like(t_matrix)
    for i in range(128):
        if d_bits[i, 0] == 1:
            q_matrix[i] = t_matrix[i] ^ s_base
        else:
            q_matrix[i] = t_matrix[i]

    # 3. Inputs
    # u (Sender input)
    u_val = np.ones((N, 2), dtype=np.uint64)
    u_val[:, 1] = 0  # u=1

    # 4. Execute Sender Round (Single Threaded Interpreter)
    print("Running Sender Round...")
    interp = Interpreter()

    # Define EDSL function to run
    def run_sender(t, s, u):
        return vole._sender_round(t, s, u, N)

    # Run
    # Wrap inputs in TensorValue? Interpreter might handle auto-wrap if we pass to run?
    # Interpreter.run(fn, *args) calls fn(*args). If args are Values, fine.
    # The Primitives `run_jax` etc expect Values.
    # We should define the function to accept Values or handle wrapping.
    # But EDSL functions expect Objects (Tracer).
    # Wait, Interpreter executes the Abstract Evaluation then Impl.
    # Actually, `Interpreter.run` does NOT Trace. It executes eagerly.
    # So we pass Values (TensorValue).

    # Create Interpreter instance
    interp = Interpreter()

    import mplang.v2.edsl.typing as elt
    from mplang.v2.runtime.interpreter import InterpObject

    # Types
    # T: (128, 128) u8
    # S: (128,) u8
    # U: (N, 2) u64

    t_type = elt.Tensor[elt.u8, (128, 128)]
    s_type = elt.Tensor[elt.u8, (128,)]
    u_type = elt.Tensor[elt.u64, (N, 2)]

    with interp:
        # Wrappers
        # Note: Runtime object for Tensor dialect is TensorValue
        t_tv = TensorValue(jnp.array(t_matrix))
        s_tv = TensorValue(jnp.array(s_base))
        u_tv = TensorValue(jnp.array(u_val))

        t_obj = InterpObject(t_tv, t_type, interp)
        s_obj = InterpObject(s_tv, s_type, interp)
        u_obj = InterpObject(u_tv, u_type, interp)

        # Run
        # Sender returns (m_corr, v_sender)
        m_tv, v_tv = run_sender(t_obj, s_obj, u_obj)

        # 5. Transfer M to Receiver
        # m_tv and v_tv are InterpObjects
        m_io = m_tv
        v_io = v_tv

        # Extract values for validation later
        # runtime_obj is TensorValue for Tensor dialect
        v = v_io.runtime_obj.unwrap()

        print("Running Receiver Round...")

        # 6. Execute Receiver Round
        def run_recv(q, m, d):
            return vole._recv_round(q, m, d, N)

        q_tv = TensorValue(jnp.array(q_matrix))
        d_tv = TensorValue(jnp.array(d_bits))

        q_type = elt.Tensor[elt.u8, (128, 128)]
        d_type = elt.Tensor[elt.u8, (128, 1)]  # bits

        q_obj = InterpObject(q_tv, q_type, interp)
        d_obj = InterpObject(d_tv, d_type, interp)

        # m_io is already an InterpObject.
        # We can pass it directly to Receiver.

        w_io = run_recv(q_obj, m_io, d_obj)

        w = w_io.runtime_obj.unwrap()

    # 7. Verification
    print("Verifying...")
    # w = v + u * delta
    # Use field mul impl
    from mplang.v2.dialects.field import _gf128_mul_impl

    mismatch = 0
    for i in range(N):
        prod = _gf128_mul_impl(u_val[i], delta_val)
        rhs = v[i] ^ prod
        if not np.array_equal(w[i], rhs):
            print(f"Mismatch at {i}: w={w[i]}, expected={rhs}, v={v[i]}")
            mismatch += 1

    if mismatch == 0:
        print("SUCCESS: Logic Verified (w = v + u*delta)")
    else:
        print(f"FAILURE: {mismatch} mismatches")
        exit(1)


if __name__ == "__main__":
    prove_logic()
