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

import mplang
import mplang.device as mpd
from mplang import ClusterSpec, Simulator, TensorType
from mplang.simp import P0, P1, P2, P2P

cluster_spec = ClusterSpec.from_dict({
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
        "TEE0": {"kind": "TEE", "members": ["node_2"], "config": {}},
    },
})


@mpd.function
def millionaire_device():
    x = mpd.device("P0")(random.randint)(0, 100)
    y = mpd.device("P1")(random.randint)(0, 100)
    # Compare at TEE with transparent PPU->TEE encryption
    z = mpd.device("TEE0")(lambda a, b: a < b)(x, y)
    # Bring result back to P0 (TEE->PPU transparent encryption)
    r = mpd.put("P0", z)
    return x, y, z, r


@mpd.function
def millionaire_manual():
    # Inputs at data parties
    x = mpd.device("P0")(random.randint)(0, 100)
    y = mpd.device("P1")(random.randint)(0, 100)

    info = "mplang/device/tee/v1"

    # P0 <-> TEE handshake and transfer x (using sugar)
    tee_sk0, tee_pk0 = P2.crypto.kem_keygen("x25519")
    quote0 = P2.tee.quote(tee_pk0)
    tee_pk0_at_p0 = P0.tee.attest(P2P(P2, P0, quote0))
    v_sk0, v_pk0 = P0.crypto.kem_keygen("x25519")
    shared0_p = P0.crypto.kem_derive(v_sk0, tee_pk0_at_p0, "x25519")
    shared0_t = P2.crypto.kem_derive(tee_sk0, P2P(P0, P2, v_pk0), "x25519")
    sess0_p = P0.crypto.hkdf(shared0_p, info)
    sess0_t = P2.crypto.hkdf(shared0_t, info)
    out_ty_x = TensorType.from_obj(x)
    bx = P0.crypto.pack(x)
    cx = P0.crypto.enc(bx, sess0_p)
    cx_at_tee = P2P(P0, P2, cx)
    bx_at_tee = P2.crypto.dec(cx_at_tee, sess0_t)
    x_at_tee = P2.crypto.unpack(bx_at_tee, out_ty_x)

    # P1 <-> TEE handshake and transfer y (still show original style for contrast)
    tee_sk1, tee_pk1 = P2.crypto.kem_keygen("x25519")
    quote1 = P2.tee.quote(tee_pk1)
    tee_pk1_at_p1 = P1.tee.attest(P2P(P2, P1, quote1))
    v_sk1, v_pk1 = P1.crypto.kem_keygen("x25519")
    shared1_p = P1.crypto.kem_derive(v_sk1, tee_pk1_at_p1, "x25519")
    shared1_t = P2.crypto.kem_derive(tee_sk1, P2P(P1, P2, v_pk1), "x25519")
    sess1_p = P1.crypto.hkdf(shared1_p, info)
    sess1_t = P2.crypto.hkdf(shared1_t, info)
    out_ty_y = TensorType.from_obj(y)
    by = P1.crypto.pack(y)
    cy = P1.crypto.enc(by, sess1_p)
    cy_at_tee = P2P(P1, P2, cy)
    by_at_tee = P2.crypto.dec(cy_at_tee, sess1_t)
    y_at_tee = P2.crypto.unpack(by_at_tee, out_ty_y)

    # Compute at TEE and send result back to P0
    z_at_tee = P2(lambda a, b: a < b, x_at_tee, y_at_tee)
    out_ty_z = TensorType.from_obj(z_at_tee)
    bz = P2.crypto.pack(z_at_tee)
    cz = P2.crypto.enc(bz, sess0_t)
    cz_at_p0 = P2P(P2, P0, cz)
    bz_at_p0 = P0.crypto.dec(cz_at_p0, sess0_p)
    r_at_p0 = P0.crypto.unpack(bz_at_p0, out_ty_z)

    return x, y, z_at_tee, r_at_p0


def main():
    print("-" * 10, "TEE millionaire: device vs manual (end-to-end IR)", "-" * 10)
    sim = Simulator(cluster_spec)

    compiled_dev = mplang.compile(sim, millionaire_device)
    compiled_man = mplang.compile(sim, millionaire_manual)
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
    xd, yd, zd, rd = mplang.evaluate(sim, millionaire_device)
    xm, ym, zm, rm = mplang.evaluate(sim, millionaire_manual)
    print(
        "device: x, y, z@TEE, r@P0 ->",
        mplang.fetch(sim, xd),
        mplang.fetch(sim, yd),
        mplang.fetch(sim, zd),
        mplang.fetch(sim, rd),
    )
    print(
        "manual: x, y, z@TEE, r@P0 ->",
        mplang.fetch(sim, xm),
        mplang.fetch(sim, ym),
        mplang.fetch(sim, zm),
        mplang.fetch(sim, rm),
    )


if __name__ == "__main__":
    main()
