"""Tests for SPU backend implementation."""

import numpy as np

import mplang2.backends.simp_simulator
import mplang2.backends.simp_worker
import mplang2.backends.spu_impl
import mplang2.backends.tensor_impl  # noqa: F401
import mplang2.edsl as el
from mplang2.backends.simp_simulator import SimpSimulator
from mplang2.dialects import simp, spu


def test_spu_e2e_simulation():
    """Test SPU end-to-end flow using SimpSimulator."""
    # 1. Setup
    world_size = 3
    sim = SimpSimulator(world_size=world_size)
    spu_device = spu.SPUDevice(parties=(0, 1, 2))

    # 2. Define computation
    def secure_add(x, y):
        return x + y

    # 3. Define workflow graph
    with el.Tracer() as tracer:
        # Create data on party 0
        # We use pcall to create MP-typed data on party 0
        x_mp = simp.constant((0,), np.array([1.0, 2.0], dtype=np.float32))
        y_mp = simp.constant((0,), np.array([3.0, 4.0], dtype=np.float32))

        # Encrypt (Public -> SPU)
        x_enc = spu.encrypt(x_mp, spu_device)
        y_enc = spu.encrypt(y_mp, spu_device)

        # Execute (SPU -> SPU)
        z_enc = spu.call(secure_add, spu_device.parties, x_enc, y_enc)

        # Decrypt (SPU -> Public)
        z_mp = spu.decrypt(z_enc, target_party=0)

        # Return result
        tracer.finalize(z_mp)

    graph = tracer.graph

    try:
        # 4. Execute on all parties
        futures = []
        for rank in range(world_size):
            # Type hint for _submit expects Operation, but we pass Graph.
            # This works at runtime because WorkerInterpreter accepts Graph.
            futures.append(sim._submit(rank, graph, {}))  # type: ignore

        results = sim._collect(futures)

        # 5. Verify
        # Result from party 0 should be the tensor
        res_p0 = results[0]

        assert res_p0 is not None
        np.testing.assert_allclose(res_p0, [4.0, 6.0])

        assert results[1] is None
        assert results[2] is None

    finally:
        sim.shutdown(wait=False)
