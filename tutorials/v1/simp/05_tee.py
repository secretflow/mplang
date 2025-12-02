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
Tutorial 9: TEE transparent encryption — device vs manual (end-to-end IR).

We implement the classic millionaire comparison in two styles and compare their MPIR:
- Device-oriented: automatic PPU↔TEE attestation + session + bytes-only enc/dec.
- Manual simp: explicit quote/attest, KEM(+HKDF), pack/enc/p2p/dec/unpack.

Security note: crypto is mock for demo; see backend/crypto warnings.
"""

from __future__ import annotations

import random

import mplang.v1 as mp
from mplang import P0, P1, P2, P2P

cluster_spec = mp.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61920"},
        {"name": "node_1", "endpoint": "127.0.0.1:61921"},
        {"name": "node_2", "endpoint": "127.0.0.1:61922"},
    ],
    "devices": {
        # SPU is not used in this tutorial, include it to make simulator happy.
        "SP0": {
            "kind": "SPU",
            "members": ["node_0", "node_1", "node_2"],
            "config": {"protocol": "SEMI2K", "field": "FM128"},
        },
        "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
        "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
        "TEE0": {
            "kind": "TEE",
            "members": ["node_2"],
        },
    },
})


@mp.function
def millionaire_device():
    x = mp.device("P0")(random.randint)(0, 100)
    y = mp.device("P1")(random.randint)(0, 100)
    # Compare at TEE with transparent PPU->TEE encryption
    z = mp.device("TEE0")(lambda a, b: a < b)(x, y)
    # Bring result back to P0 (TEE->PPU transparent encryption)
    r = mp.put("P0", z)
    return x, y, z, r


@mp.function
def millionaire_manual():
    # Inputs at data parties
    x = mp.device("P0")(random.randint)(0, 100)
    y = mp.device("P1")(random.randint)(0, 100)

    info = "mplang/device/tee/v1"

    # P0 <-> TEE handshake and transfer x (using sugar)
    tee_sk0, tee_pk0 = P2.crypto.kem_keygen("x25519")
    quote0 = P2.tee.quote_gen(tee_pk0)
    tee_pk0_at_p0 = P0.tee.attest(P2P(P2, P0, quote0))
    v_sk0, v_pk0 = P0.crypto.kem_keygen("x25519")
    shared0_p = P0.crypto.kem_derive(v_sk0, tee_pk0_at_p0, "x25519")
    shared0_t = P2.crypto.kem_derive(tee_sk0, P2P(P0, P2, v_pk0), "x25519")
    sess0_p = P0.crypto.hkdf(shared0_p, info)
    sess0_t = P2.crypto.hkdf(shared0_t, info)
    out_ty_x = mp.TensorType.from_obj(x)
    bx = P0.basic.pack(x)
    cx = P0.crypto.enc(bx, sess0_p)
    cx_at_tee = P2P(P0, P2, cx)
    bx_at_tee = P2.crypto.dec(cx_at_tee, sess0_t)
    x_at_tee = P2.basic.unpack(bx_at_tee, out_ty_x)

    # P1 <-> TEE handshake and transfer y (still show original style for contrast)
    tee_sk1, tee_pk1 = P2.crypto.kem_keygen("x25519")
    quote1 = P2.tee.quote_gen(tee_pk1)
    tee_pk1_at_p1 = P1.tee.attest(P2P(P2, P1, quote1))
    v_sk1, v_pk1 = P1.crypto.kem_keygen("x25519")
    shared1_p = P1.crypto.kem_derive(v_sk1, tee_pk1_at_p1, "x25519")
    shared1_t = P2.crypto.kem_derive(tee_sk1, P2P(P1, P2, v_pk1), "x25519")
    sess1_p = P1.crypto.hkdf(shared1_p, info)
    sess1_t = P2.crypto.hkdf(shared1_t, info)
    out_ty_y = mp.TensorType.from_obj(y)
    by = P1.basic.pack(y)
    cy = P1.crypto.enc(by, sess1_p)
    cy_at_tee = P2P(P1, P2, cy)
    by_at_tee = P2.crypto.dec(cy_at_tee, sess1_t)
    y_at_tee = P2.basic.unpack(by_at_tee, out_ty_y)

    # Compute at TEE and send result back to P0
    z_at_tee = P2(lambda a, b: a < b, x_at_tee, y_at_tee)
    out_ty_z = mp.TensorType.from_obj(z_at_tee)
    bz = P2.basic.pack(z_at_tee)
    cz = P2.crypto.enc(bz, sess0_t)
    cz_at_p0 = P2P(P2, P0, cz)
    bz_at_p0 = P0.crypto.dec(cz_at_p0, sess0_p)
    r_at_p0 = P0.basic.unpack(bz_at_p0, out_ty_z)

    return x, y, z_at_tee, r_at_p0


@mp.function
def millionaire_tee_spu():
    """
    Demonstrates TEE <-> SPU data transmission using device API.
    Follows the same pattern as millionaire_device but adds SPU step.
    """
    a = mp.device("P0")(random.randint)(0, 100)
    b = mp.device("P1")(random.randint)(0, 100)

    # Step 1: Process at TEE (like millionaire_device does comparison)
    a_processed = mp.device("TEE0")(lambda v: v * 2)(a)
    b_processed = mp.device("TEE0")(lambda v: v + 10)(b)

    # Step 2: Transfer processed results from TEE to SPU
    a_at_spu = mp.put("SP0", a_processed)
    b_at_spu = mp.put("SP0", b_processed)

    # Step 3: Secure computation at SPU
    c_at_spu = mp.device("SP0")(lambda a, b: a < b)(a_at_spu, b_at_spu)

    # Step 4: Transfer result from SPU back to TEE
    c_at_tee = mp.put("TEE0", c_at_spu)

    # Step 5: Final processing at TEE and return to P0 (like millionaire_device)
    final_result = mp.device("TEE0")(lambda v: v + 1000)(c_at_tee)
    result_at_p0 = mp.put("P0", final_result)

    return a, b, a_processed, b_processed, c_at_spu, final_result, result_at_p0


def main():
    print("-" * 10, "TEE millionaire: device vs manual (end-to-end IR)", "-" * 10)
    # Create simulator with TEE bindings
    tee_bindings = {
        "tee.quote_gen": "mock_tee.quote_gen",
        "tee.attest": "mock_tee.attest",
    }
    # Apply tee_bindings per-node (preferred) then construct Simulator
    for n in cluster_spec.nodes.values():
        n.runtime_info.op_bindings.update(tee_bindings)
    sim = mp.Simulator(cluster_spec)

    compiled_dev = mp.compile(sim, millionaire_device)
    compiled_man = mp.compile(sim, millionaire_manual)
    ir_dev = compiled_dev.compiler_ir()
    ir_man = compiled_man.compiler_ir()

    import re

    def normalize_ir(s: str) -> str:
        s = re.sub(r"\{_devid_=\"[^\"]*\"\}", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    same_raw = ir_dev == ir_man
    same_norm = normalize_ir(ir_dev) == normalize_ir(ir_man)
    print("IR equal (raw):", same_raw)
    print("IR equal (normalized):", same_norm)
    if not same_norm:
        print("\n--- Device IR ---\n", ir_dev)
        print("\n--- Manual IR ---\n", ir_man)
    print("IR:", ir_dev)

    # Run both
    xd, yd, zd, rd = mp.evaluate(sim, millionaire_device)
    xm, ym, zm, rm = mp.evaluate(sim, millionaire_manual)
    print(
        "device: x, y, z@TEE, r@P0 ->",
        mp.fetch(sim, xd),
        mp.fetch(sim, yd),
        mp.fetch(sim, zd),
        mp.fetch(sim, rd),
    )
    print(
        "manual: x, y, z@TEE, r@P0 ->",
        mp.fetch(sim, xm),
        mp.fetch(sim, ym),
        mp.fetch(sim, zm),
        mp.fetch(sim, rm),
    )

    # Test TEE-SPU data transmission step by step
    print("\n" + "-" * 10, "TEE-SPU data transmission demo", "-" * 10)

    compiled_full = mp.compile(sim, millionaire_tee_spu)
    print("millionaire_tee_spu IR:", compiled_full.compiler_ir())
    a, b, a_proc, b_proc, c_spu, final, result = mp.evaluate(sim, millionaire_tee_spu)
    print(
        "✓ Full TEE-SPU successful: a@P0, b@P1, a_proc@TEE, b_proc@TEE, c@SPU, final@TEE, result@P0 ->",
        mp.fetch(sim, a),
        mp.fetch(sim, b),
        mp.fetch(sim, a_proc),
        mp.fetch(sim, b_proc),
        mp.fetch(sim, c_spu),
        mp.fetch(sim, final),
        mp.fetch(sim, result),
    )


if __name__ == "__main__":
    main()
