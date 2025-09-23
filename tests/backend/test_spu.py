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

"""SPU flat backend kernel tests.

Rewritten to remove deprecated SpuHandler usage. These tests exercise the
registered kernels:
    - spu.makeshares
    - spu.reconstruct
    - mlir.pphlo (compiled via frontend.spu.jax_compile)

We keep the tests intentionally lean: just enough coverage to ensure the new
flat kernel pathway functions for single-party (REF2K) and multi-party (SEMI2K)
execution flows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp
import numpy as np
import spu.libspu as libspu

from mplang.backend.base import (
    create_runtime,
    list_registered_kernels,
)
from mplang.backend.spu import SpuValue
from mplang.core.cluster import ClusterSpec, Device, Node, RuntimeInfo
from mplang.core.dtype import DType
from mplang.core.mpobject import MPContext, MPObject
from mplang.core.mptype import MPType
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType
from mplang.frontend import spu
from mplang.runtime.link_comm import LinkCommunicator


class DummyContext(MPContext):
    """Minimal MPContext for compiling JAX functions to SPU PFunction."""

    def __init__(self):
        runtime = RuntimeInfo(version="dev", platform="local", backends=[])
        node = Node(name="p0", rank=0, endpoint="local", runtime_info=runtime)
        device = Device(name="p0_local", kind="local", members=[node])
        spec = ClusterSpec(nodes={node.name: node}, devices={device.name: device})
        super().__init__(spec)


class DummyTensor(MPObject):
    """Minimal MPObject used purely for spu.jax_compile shape/dtype inference."""

    def __init__(self, shape: tuple[int, ...], dtype: jnp.dtype):  # type: ignore[valid-type]
        self._mptype = MPType.tensor(DType.from_any(dtype), shape)
        self._ctx = DummyContext()

    @property
    def mptype(self) -> MPType:  # pragma: no cover - trivial
        return self._mptype

    @property
    def ctx(self) -> MPContext:  # pragma: no cover - trivial
        return self._ctx


def create_mem_link_contexts(world_size: int) -> list[LinkCommunicator]:
    """Create memory link communicators for each party.

    Returns a list so tests can still iterate, but initialize_spu_runtime will
    now be called per-rank with a single link instance.
    """
    addrs = [f"P{i}" for i in range(world_size)]
    return [LinkCommunicator(rank, addrs, mem_link=True) for rank in range(world_size)]


def _compile_add(shape: tuple[int, ...] = (3,), dtype=jnp.float32):  # type: ignore[valid-type]
    def add_fn(x, y):
        return x + y

    args = [DummyTensor(shape, dtype), DummyTensor(shape, dtype)]
    pfunc, _ins, _out_tree = spu.jax_compile(add_fn, *args)
    return pfunc


def _makeshares_pfunc(arr: np.ndarray, world_size: int) -> PFunction:
    """Create makeshares PFunction with outs matching expected party world size.

    We rely on caller providing world_size consistent with runtime creation config.
    """
    tensor_type = TensorType.from_obj(arr)
    outs = tuple(tensor_type for _ in range(world_size))
    return PFunction(
        fn_type="spu.makeshares",
        ins_info=(tensor_type,),
        outs_info=outs,
        visibility=libspu.Visibility.VIS_SECRET.value,
    )


def _reconstruct_pfunc(example: np.ndarray, world_size: int) -> PFunction:
    """Create a reconstruct PFunction whose ins_info matches world size.

    Each input is a share with same tensor type meta; backend kernel validates
    actual share objects are SpuValue.
    """
    tensor_type = TensorType.from_obj(example)
    return PFunction(
        fn_type="spu.reconstruct",
        ins_info=tuple(tensor_type for _ in range(world_size)),
        outs_info=(tensor_type,),
    )


class TestSpuKernels:
    def test_kernel_registry(self):
        for name in ["spu.makeshares", "spu.reconstruct", "mlir.pphlo"]:
            assert name in list_registered_kernels()

    def test_makeshares_reconstruct_single_party(self):
        world = 1
        cfg = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.REF2K, field=libspu.FieldType.FM128
        )
        link_ctxs = create_mem_link_contexts(world)
        runtime = create_runtime(0, world)
        # Seed SPU env via kernel
        seed_fn = PFunction(
            fn_type="spu.seed_env",
            ins_info=(),
            outs_info=(),
            config=cfg,
            world=world,
            link=link_ctxs[0],
        )
        runtime.run_kernel(seed_fn, [])

        x = np.array([1, 2, 3], dtype=np.int32)
        mk = _makeshares_pfunc(x, world)
        shares = runtime.run_kernel(mk, [x])
        assert len(shares) == 1 and isinstance(shares[0], SpuValue)
        rc = _reconstruct_pfunc(x, world)
        # run_kernel expects outs length=1; we call reconstruct with the share list
        out = runtime.run_kernel(rc, shares)[0]
        np.testing.assert_array_equal(out, x)

    def test_makeshares_reconstruct_multiparty(self):
        world = 2
        cfg = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.SEMI2K, field=libspu.FieldType.FM128
        )
        link_ctxs = create_mem_link_contexts(world)
        # seed rank0 state
        runtime0 = create_runtime(0, world)
        seed0 = PFunction(
            fn_type="spu.seed_env",
            ins_info=(),
            outs_info=(),
            config=cfg,
            world=world,
            link=link_ctxs[0],
        )
        runtime0.run_kernel(seed0, [])
        # seed rank1 state to simulate multi-rank kernel contexts
        runtime1 = create_runtime(1, world)
        seed1 = PFunction(
            fn_type="spu.seed_env",
            ins_info=(),
            outs_info=(),
            config=cfg,
            world=world,
            link=link_ctxs[1],
        )
        runtime1.run_kernel(seed1, [])

        x = np.array([10, 20], dtype=np.int32)
        mk = _makeshares_pfunc(x, world)
        shares = runtime0.run_kernel(mk, [x])
        assert len(shares) == world and all(isinstance(s, SpuValue) for s in shares)
        rc = _reconstruct_pfunc(x, world)
        out = runtime0.run_kernel(rc, list(shares))[0]
        np.testing.assert_array_equal(out, x)

    def test_mlir_pphlo_single_party(self):
        world = 1
        cfg = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.REF2K, field=libspu.FieldType.FM128
        )
        link_ctxs = create_mem_link_contexts(world)
        runtime = create_runtime(0, world)
        seed = PFunction(
            fn_type="spu.seed_env",
            ins_info=(),
            outs_info=(),
            config=cfg,
            world=world,
            link=link_ctxs[0],
        )
        runtime.run_kernel(seed, [])

        pfunc = _compile_add((3,), jnp.float32)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = x + y

        mkx = _makeshares_pfunc(x, world)
        mky = _makeshares_pfunc(y, world)
        x_shares = runtime.run_kernel(mkx, [x])
        y_shares = runtime.run_kernel(mky, [y])

        # Single-party run (rank=0)
        result_share = runtime.run_kernel(pfunc, [x_shares[0], y_shares[0]])[0]
        assert isinstance(result_share, SpuValue)
        rc = _reconstruct_pfunc(expected, world)
        out = runtime.run_kernel(rc, [result_share])[0]
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_mlir_pphlo_multiparty(self):
        world = 2
        cfg = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.SEMI2K, field=libspu.FieldType.FM128
        )
        link_ctxs = create_mem_link_contexts(world)

        # Initialize per-rank runtime & seed (store explicit runtimes)
        runtimes = {}
        for r in range(world):
            rt = create_runtime(r, world)
            runtimes[r] = rt
            seed_fn = PFunction(
                fn_type="spu.seed_env",
                ins_info=(),
                outs_info=(),
                config=cfg,
                world=world,
                link=link_ctxs[r],
            )
            rt.run_kernel(seed_fn, [])  # use explicit runtime

        pfunc = _compile_add((3,), jnp.float32)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = x + y

        mkx = _makeshares_pfunc(x, world)
        mky = _makeshares_pfunc(y, world)
        x_shares: list[SpuValue] = runtimes[0].run_kernel(mkx, [x])
        y_shares: list[SpuValue] = runtimes[0].run_kernel(mky, [y])

        # Run mlir.pphlo concurrently per rank to satisfy interactive protocol
        def party(rank: int, xs: SpuValue, ys: SpuValue):
            rt = runtimes[rank]
            return rt.run_kernel(pfunc, [xs, ys])[0]

        with ThreadPoolExecutor(max_workers=world) as pool:
            futures = [
                pool.submit(party, r, x_shares[r], y_shares[r]) for r in range(world)
            ]
            results = [f.result() for f in futures]

        assert len(results) == world and all(isinstance(r, SpuValue) for r in results)
        rc = _reconstruct_pfunc(expected, world)
        out = runtimes[0].run_kernel(rc, results)[0]
        np.testing.assert_allclose(out, expected, rtol=1e-5)
