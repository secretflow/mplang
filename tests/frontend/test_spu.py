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

import jax.numpy as jnp
import pytest
import spu.libspu as libspu

from mplang.core.cluster import ClusterSpec, Device, Node, RuntimeInfo
from mplang.core.dtype import DType
from mplang.core.mpobject import MPContext, MPObject
from mplang.core.mptype import MPType
from mplang.frontend.spu import (
    SpuConfig,
    SpuJaxCompile,
    SpuMakeShares,
    SpuReconstruct,
    Visibility,
)


class DummyContext(MPContext):
    def __init__(self) -> None:
        runtime = RuntimeInfo(version="dev", platform="local", backends=[])
        node = Node(name="p0", rank=0, endpoint="local", runtime_info=runtime)
        device = Device(name="p0_local", kind="local", members=[node])
        spec = ClusterSpec(nodes={node.name: node}, devices={device.name: device})
        super().__init__(spec)


class Tensor(MPObject):
    """Minimal tensor MPObject for SPU frontend tests."""

    def __init__(self, arr: jnp.ndarray) -> None:  # type: ignore[valid-type]
        # jax arrays expose .dtype and .shape
        self._arr = arr
        self._mptype = MPType.tensor(DType.from_any(str(arr.dtype)), tuple(arr.shape))
        self._ctx = DummyContext()

    @property
    def mptype(self) -> MPType:
        return self._mptype

    @property
    def ctx(self) -> MPContext:
        return self._ctx

    def runtime_obj(self) -> jnp.ndarray:  # type: ignore[valid-type]
        return self._arr


def test_make_shares_basic() -> None:
    cfg = SpuConfig(world_size=3)
    op = SpuMakeShares(cfg)
    t = Tensor(jnp.array([1, 2, 3], dtype=jnp.float32))
    pfunc, ins, _out_tree = op(t)
    assert pfunc.fn_type == "spu.makeshares"
    assert len(ins) == 1
    assert len(pfunc.outs_info) == 3
    assert pfunc.attrs["world_size"] == 3


def test_make_shares_private_validation() -> None:
    cfg = SpuConfig(world_size=3, enable_private=True)
    op = SpuMakeShares(cfg)
    t = Tensor(jnp.array([0], dtype=jnp.int32))
    pfunc, _, _ = op(t, visibility=Visibility.PRIVATE, owner_rank=2)
    assert pfunc.attrs["owner_rank"] == 2
    assert pfunc.attrs["visibility"] == libspu.Visibility.VIS_PRIVATE


def test_reconstruct_basic() -> None:
    cfg = SpuConfig(world_size=2)
    recon = SpuReconstruct(cfg)
    s1 = Tensor(jnp.array([1, 2]))
    s2 = Tensor(jnp.array([3, 4]))
    pfunc, ins, _tree = recon(s1, s2)
    assert pfunc.fn_type == "spu.reconstruct"
    assert len(ins) == 2
    assert len(pfunc.outs_info) == 1


def test_reconstruct_world_size_mismatch() -> None:
    cfg = SpuConfig(world_size=3)
    recon = SpuReconstruct(cfg)
    s1 = Tensor(jnp.array([1]))
    s2 = Tensor(jnp.array([2]))
    with pytest.raises(ValueError):
        recon(s1, s2)  # only 2 shares, expect 3


def test_jax_compile_simple_add() -> None:
    cfg = SpuConfig(world_size=3)
    compiler = SpuJaxCompile(cfg)

    def fn(a, b):  # type: ignore[no-untyped-def]
        return a + b

    a = Tensor(jnp.array([1.0, 2.0], dtype=jnp.float32))
    b = Tensor(jnp.array([3.0, 4.0], dtype=jnp.float32))
    pfunc, ins, _out_tree = compiler(fn, a, b)
    assert pfunc.fn_type == "mlir.pphlo"
    assert len(ins) == 2
    assert len(pfunc.outs_info) == 1
    assert pfunc.fn_text is not None and "func" in pfunc.fn_text


def test_jax_compile_multiple_outputs() -> None:
    cfg = SpuConfig(world_size=3)
    compiler = SpuJaxCompile(cfg)

    def fn(a):  # type: ignore[no-untyped-def]
        return a + 1, a * 2

    a = Tensor(jnp.array([1, 2, 3], dtype=jnp.int32))
    pfunc, _ins, _out_tree = compiler(fn, a)
    assert len(pfunc.outs_info) == 2


def test_jax_compile_visibility_metadata_secret() -> None:
    cfg = SpuConfig(world_size=3)
    compiler = SpuJaxCompile(cfg)

    def fn(a):  # type: ignore[no-untyped-def]
        return a * 2

    a = Tensor(jnp.array([1, 2], dtype=jnp.float32))
    pfunc, _ins, _ = compiler(fn, a)
    assert pfunc.attrs["input_visibilities"] == [libspu.Visibility.VIS_SECRET]
