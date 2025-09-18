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

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import jax.numpy as jnp
import spu.libspu as libspu
import spu.utils.frontend as spu_fe
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction, get_fn_name
from mplang.core.tensor import TensorType
from mplang.frontend.base import femod
from mplang.utils.func_utils import normalize_fn


class Visibility(Enum):
    """Visibility types for SPU shares."""

    SECRET = libspu.Visibility.VIS_SECRET
    PUBLIC = libspu.Visibility.VIS_PUBLIC
    PRIVATE = libspu.Visibility.VIS_PRIVATE


_SPU_MOD = femod("spu")


@_SPU_MOD.feop()
def makeshares(
    data: MPObject,
    *,
    world_size: int,
    visibility: Visibility = Visibility.SECRET,
    owner_rank: int = -1,
    enable_private: bool = False,
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Create SPU shares from a plaintext tensor.

    SSOT module-function version. Output shares count equals world_size.
    """
    if not isinstance(data, MPObject):
        raise TypeError("data must be an MPObject (Tensor) for makeshares")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if visibility == Visibility.PRIVATE:
        if not enable_private:
            raise ValueError("PRIVATE visibility disabled; set enable_private=True")
        if owner_rank < 0 or owner_rank >= world_size:
            raise ValueError(f"owner_rank {owner_rank} out of range [0,{world_size})")

    in_ty = TensorType.from_obj(data)
    ins_info = (in_ty,)
    outs_info = tuple(in_ty for _ in range(world_size))
    pfunc = PFunction(
        fn_type="spu.makeshares",
        ins_info=ins_info,
        outs_info=outs_info,
        fn_name="makeshares",
        visibility=visibility.value,
        owner_rank=owner_rank,
        operation="makeshares",
        world_size=world_size,
    )
    _, treedef = tree_flatten(list(outs_info))
    return pfunc, [data], treedef


@_SPU_MOD.feop()
def reconstruct(
    *shares: MPObject,
    world_size: int | None = None,
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Reconstruct plaintext tensor from shares.

    If world_size is provided, enforce matching share count; otherwise infer from inputs.
    """
    if len(shares) == 0:
        raise ValueError("reconstruct requires at least one share")
    if world_size is not None and len(shares) != world_size:
        raise ValueError(f"Expected {world_size} shares, got {len(shares)}")

    ins_info = tuple(TensorType.from_obj(s) for s in shares)
    outs_info = (ins_info[0],)
    pfunc = PFunction(
        fn_type="spu.reconstruct",
        ins_info=ins_info,
        outs_info=outs_info,
        fn_name="reconstruct",
    )
    _, treedef = tree_flatten(outs_info[0])
    return pfunc, list(shares), treedef


@_SPU_MOD.feop()
def jax_compile(
    fn: Callable, *args: Any, copts: Any | None = None, **kwargs: Any
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Compile a JAX function into SPU pphlo MLIR representation."""

    def is_variable(arg: Any) -> bool:
        return isinstance(arg, MPObject)

    normalized_fn, in_vars = normalize_fn(fn, args, kwargs, is_variable)

    jax_params = [
        ShapeDtypeStruct(arg.shape, jnp.dtype(arg.dtype.name)) for arg in in_vars
    ]
    in_vis = [libspu.Visibility.VIS_SECRET for _ in in_vars]
    in_names = [f"in{idx}" for idx in range(len(in_vars))]
    out_names_gen = lambda outs: [f"out{idx}" for idx in range(len(outs))]

    if copts is None:
        copts = libspu.CompilerOptions()

    executable, out_info = spu_fe.compile(
        spu_fe.Kind.JAX,
        normalized_fn,
        [jax_params],
        {},
        in_names,
        in_vis,
        out_names_gen,
        static_argnums=(),
        static_argnames=None,
        copts=copts,
    )
    out_info_flat, out_tree = tree_flatten(out_info)
    output_tensor_infos = [TensorType.from_obj(out) for out in out_info_flat]

    executable_code = executable.code
    assert isinstance(executable_code, bytes), (
        f"Expected bytes, got {type(executable_code)}"
    )
    executable_code = executable_code.decode("utf-8")

    pfunc = PFunction(
        fn_type="mlir.pphlo",
        ins_info=tuple(TensorType.from_obj(x) for x in in_vars),
        outs_info=tuple(output_tensor_infos),
        fn_name=get_fn_name(fn),
        fn_text=executable_code,
        input_visibilities=in_vis,
        input_names=list(executable.input_names),
        output_names=list(executable.output_names),
        executable_name=executable.name,
    )
    return pfunc, in_vars, out_tree
