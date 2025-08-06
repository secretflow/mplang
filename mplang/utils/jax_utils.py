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

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import jax
from jax.lib import xla_extension as xla

jax.config.update("jax_enable_x64", True)


@dataclass
class JittedFunction(ABC):
    fn: Callable

    @abstractmethod
    def trace(self, *args, **kwargs) -> JittedFunction: ...

    @abstractmethod
    def as_hlo_module(self) -> xla.HloModule: ...

    @abstractmethod
    def out_info(self): ...


def _argnames_partial_except(fn, static_argnames, kwargs):
    if static_argnames is None:
        return fn, kwargs

    assert isinstance(
        static_argnames, (str, Iterable)
    ), f"type of static_argnames is {type(static_argnames)} while str or Iterable is required here."
    if isinstance(static_argnames, str):
        static_argnames = (static_argnames,)

    static_kwargs = {k: kwargs.pop(k) for k in static_argnames if k in kwargs}
    return functools.partial(fn, **static_kwargs), kwargs


@dataclass
class JaxFunction(JittedFunction):
    static_argnums: int | Sequence[int] = ()
    static_argnames: str | Iterable[str] | None = None

    _cfn = None
    _out_info = None

    # override
    def trace(self, *args, **kwargs) -> JaxFunction:
        import jax
        from jax._src.lib import xla_extension_version
        from jax._src.xla_bridge import (
            _backend_lock,
            _backends,
            register_backend_factory,
        )

        # Register interpreter backend since we don't want any cpu/gpu/tpu specific optimization
        if xla_extension_version < 164:
            # interpreter is registerd by default before jaxlib 0.4.13
            pass
        else:
            has_interpreter_backend = False
            with _backend_lock:
                if "interpreter" in _backends:
                    has_interpreter_backend = True
            if not has_interpreter_backend:
                if xla_extension_version < 194:
                    from jax.lib import xla_client as xc

                    # make_interpreter_client has been removed after jaxlib 0.4.16
                    register_backend_factory(
                        "interpreter", xc.make_interpreter_client, priority=-100
                    )
                else:
                    from jax.interpreters.xla import Backend as xla_back

                    register_backend_factory("interpreter", xla_back, priority=-100)

        jax_version = jax.__version_info__

        if jax_version[0] > 1 or jax_version[1] > 4 or jax_version[2] > 29:
            # xla_computation is deprecated since 0.4.30, move to new api
            lowered = (
                jax.jit(
                    self.fn,
                    static_argnums=self.static_argnums,
                    static_argnames=self.static_argnames,
                    keep_unused=True,
                )
                .trace(*args, **kwargs)
                .lower(lowering_platforms=("interpreter",))
            )
            self._cfn = lowered.compiler_ir("hlo")
            self._out_info = lowered.out_info
        else:
            fn, kwargs = _argnames_partial_except(self.fn, self.static_argnames, kwargs)

            cfn, output = jax.xla_computation(
                fn,
                return_shape=True,
                static_argnums=self.static_argnums,
                backend="interpreter",
            )(*args, **kwargs)

            self._cfn = cfn
            self._out_info = output

        return self

    # override
    def as_hlo_module(self) -> xla.HloModule:
        if self._cfn is None:
            raise Exception("not traced yet.")

        # a = self._cfn.as_hlo_module().as_serialized_hlo_module_proto()
        # b = self._cfn.as_serialized_hlo_module_proto()

        return self._cfn.as_hlo_module()

    # override
    def out_info(self):
        if self._out_info is None:
            raise Exception("not traced yet.")

        return self._out_info


def jax_jit(
    fn: Callable,
    static_argnums: int | Sequence[int] = (),
    static_argnames: str | Iterable[str] | None = None,
) -> JaxFunction:
    return JaxFunction(
        fn,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )


class HloUtil:
    """The XLA hlo utilities"""

    @classmethod
    def module_to_text(cls, module: xla.HloModule):
        opts = xla.HloPrintOptions.short_parsable()
        return module.to_string(opts)

    @classmethod
    def text_to_module(cls, hlo_text: str) -> xla.HloModule:
        from jax._src.lib import xla_extension as xla_internal

        return xla_internal.hlo_module_from_text(hlo_text)

    @classmethod
    def module_to_proto(cls, module: xla.HloModule):
        return module.as_serialized_hlo_module_proto()

    @classmethod
    def proto_to_module(cls, serialized_hlo_proto: bytes) -> xla.HloModule:
        return xla.HloModule.from_serialized_hlo_module_proto(serialized_hlo_proto)

    @classmethod
    def proto_to_text(cls, serialized_hlo_proto: bytes) -> str:
        return cls.module_to_text(cls.proto_to_module(serialized_hlo_proto))

    @classmethod
    def text_to_proto(cls, text: str) -> bytes:
        return cls.module_to_proto(cls.text_to_module(text))

    @classmethod
    def text_to_mlir(cls, text: str) -> str:
        prot = cls.text_to_proto(text)
        return cls.proto_to_mlir(prot)

    @classmethod
    def proto_to_mlir(cls, serialized_hlo_proto: bytes) -> str:
        from jax._src.lib import xla_extension as xla_internal

        xla_computation = xla_internal.XlaComputation(serialized_hlo_proto)
        return xla.mlir.xla_computation_to_mlir_module(xla_computation)  # type: ignore
