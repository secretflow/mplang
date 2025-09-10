import numpy as np

import mplang
import mplang.simp as simp
from mplang.frontend import crypto, tee


def _demo_flow():
    P0, P1, P2 = 0, 1, 2
    key_a = simp.runAt(P2, crypto.keygen)(16)
    key_b = simp.runAt(P2, crypto.keygen)(16)
    quotes = simp.runAt(P2, tee.quote)([key_a, key_b])
    q0 = simp.p2p(P2, P0, quotes[0])
    q1 = simp.p2p(P2, P1, quotes[1])
    k0 = simp.runAt(P0, tee.attest)(q0)
    k1 = simp.runAt(P1, tee.attest)(q1)
    x0 = simp.runAt(P0, lambda: np.array([10, 20, 30], dtype=np.uint8))()
    x1 = simp.runAt(P1, lambda: np.array([1, 2, 3], dtype=np.uint8))()
    c0 = simp.runAt(P0, crypto.enc)(x0, k0)
    c1 = simp.runAt(P1, crypto.enc)(x1, k1)
    c0_at_tee = simp.p2p(P0, P2, c0)
    c1_at_tee = simp.p2p(P1, P2, c1)
    p0 = simp.runAt(P2, crypto.dec)(c0_at_tee, key_a)
    p1 = simp.runAt(P2, crypto.dec)(c1_at_tee, key_b)
    return p0, p1


def test_crypto_enc_dec_and_tee_quote_attest_roundtrip():
    sim = mplang.Simulator.simple(3)
    p0, p1 = mplang.evaluate(sim, _demo_flow)
    a = mplang.fetch(sim, p0)
    b = mplang.fetch(sim, p1)
    # Expect third element to be the numpy arrays propagated
    assert (a[2] == np.array([10, 20, 30], dtype=np.uint8)).all()
    assert (b[2] == np.array([1, 2, 3], dtype=np.uint8)).all()
