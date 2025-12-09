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
from typing import Any

import jax.numpy as jnp
import spu.libspu as libspu
import spu.utils.frontend as spu_fe
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.v1.core import MPObject, PFunction, TensorType, get_fn_name
from mplang.v1.ops.base import stateless_mod
from mplang.v1.utils.func_utils import normalize_fn


class Visibility:
    """Frontend visibility constants mapping to libspu.Visibility.

    Note: these are direct aliases to libspu.Visibility members so that
    downstream serialization and backends receive the exact enum type
    they expect. Keep the friendly names (SECRET/PUBLIC/PRIVATE) for
    frontend ergonomics.
    """

    SECRET = libspu.Visibility.VIS_SECRET
    PUBLIC = libspu.Visibility.VIS_PUBLIC
    PRIVATE = libspu.Visibility.VIS_PRIVATE


_SPU_MOD = stateless_mod("spu")


@_SPU_MOD.simple_op()
def makeshares(
    data: TensorType,
    *,
    world_size: int,
    visibility: libspu.Visibility = Visibility.SECRET,
    owner_rank: int = -1,
    enable_private: bool = False,
) -> tuple:
    """Create SPU shares from a plaintext tensor (type-only kernel).

    Returns a PyTree of TensorType repeated `world_size` times.
    Validation only; PFunction assembly handled by typed_op decorator.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if visibility == Visibility.PRIVATE:
        if not enable_private:
            raise ValueError("PRIVATE visibility disabled; set enable_private=True")
        if owner_rank < 0 or owner_rank >= world_size:
            raise ValueError(f"owner_rank {owner_rank} out of range [0,{world_size})")
    return tuple(data for _ in range(world_size))


@_SPU_MOD.op_def()
def reconstruct(*shares: MPObject) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Reconstruct plaintext tensor from shares."""
    if len(shares) == 0:
        raise ValueError("reconstruct requires at least one share")

    ins_info = tuple(TensorType.from_obj(s) for s in shares)
    outs_info = (ins_info[0],)
    pfunc = PFunction(
        fn_type="spu.reconstruct",
        ins_info=ins_info,
        outs_info=outs_info,
    )
    _, treedef = tree_flatten(outs_info[0])
    return pfunc, list(shares), treedef


def _compile_jax(
    copts: libspu.CompilerOptions,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """Compile a JAX function into SPU pphlo MLIR and wrap as PFunction.

    Resulting PFunction uses fn_type 'spu.run_pphlo'.
    """

    def is_variable(arg: Any) -> bool:
        return isinstance(arg, MPObject)

    normalized_fn, in_vars = normalize_fn(fn, args, kwargs, is_variable)

    jax_params = [
        ShapeDtypeStruct(arg.shape, jnp.dtype(arg.dtype.name)) for arg in in_vars
    ]
    in_vis = [libspu.Visibility.VIS_SECRET for _ in in_vars]
    in_names = [f"in{idx}" for idx in range(len(in_vars))]
    out_names_gen = lambda outs: [f"out{idx}" for idx in range(len(outs))]

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
        fn_type="spu.run_pphlo",
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


@_SPU_MOD.op_def()
def jax_compile(
    fn: Callable, *args: Any, **kwargs: Any
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    return _compile_jax(libspu.CompilerOptions(), fn, *args, **kwargs)
