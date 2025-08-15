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

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

from mplang.core.base import TensorInfo, TensorLike


class PFuncTypes:
    """Function type constants for different runtime engines."""

    MLIR_STABLEHLO = "mlir[stablehlo]"
    SPU_RUN = "spu.run"
    SPU_MAKESHARES = "spu.makeshares"
    SPU_RECONSTRUCT = "spu.reconstruct"


class PFunction:
    """A Party Function is a local function that can be executed by a single party.

    PFunction represents a computation unit that:
    1. Can be expressed and defined in the mplang frontend
    2. Can be serialized for transmission between components
    3. Can be interpreted and executed by backend runtime engines

    It provides a unified interface for describing single-party computations
    in multi-party computing scenarios, accepting a list of TensorLike inputs
    and producing a list of TensorLike outputs.

    Args:
        fn_type: The type/category of the function (e.g., "python[jax]", "builtin")
        fn_name: Human-readable name of the function
        fn_text: Optional serialized representation (e.g., MLIR, bytecode)
        ins_info: Tensor information for input parameters
        outs_info: Tensor information for output values
        attrs: Additional attributes and metadata
    """

    fn_type: str
    fn_name: str
    fn_text: str | bytes | None
    ins_info: tuple[TensorInfo, ...]
    outs_info: tuple[TensorInfo, ...]
    attrs: MappingProxyType[str, Any]

    def __init__(
        self,
        fn_type: str,
        fn_name: str,
        fn_text: str | bytes | None,
        ins_info: Sequence[TensorInfo],
        outs_info: Sequence[TensorInfo],
        attrs: dict[str, Any] | None = None,
    ):
        if attrs is None:
            attrs = {}
        self.fn_type = fn_type
        self.fn_name = fn_name
        self.fn_text = fn_text
        self.ins_info = tuple(ins_info)
        self.outs_info = tuple(outs_info)
        # Make attrs immutable to ensure PFunction immutability
        # Create a copy first, then wrap it in MappingProxyType
        if isinstance(attrs, MappingProxyType):
            self.attrs = attrs
        else:
            self.attrs = MappingProxyType(copy.copy(attrs))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.fn_type}, {self.fn_name})"

    def __hash__(self) -> int:
        return hash((
            self.fn_type,
            self.fn_name,
            self.fn_text,
            self.ins_info,
            self.outs_info,
            frozenset(self.attrs.items()),
        ))

    def __eq__(self, other: object) -> bool:
        """Check equality between PFunction instances."""
        if not isinstance(other, PFunction):
            return False

        return (
            self.fn_type == other.fn_type
            and self.fn_name == other.fn_name
            and self.fn_text == other.fn_text
            and self.ins_info == other.ins_info
            and self.outs_info == other.outs_info
            and self.attrs == other.attrs
        )


def get_fn_name(fn_like: Any) -> str:
    if hasattr(fn_like, "__name__"):
        return fn_like.__name__  # type: ignore[no-any-return]
    if hasattr(fn_like, "func"):
        # handle partial functions
        return get_fn_name(fn_like.func)
    return "unnamed function"


class PFunctionHandler(ABC):
    """The base class for PFunction Handlers."""

    @abstractmethod
    def list_fn_names(self) -> list[str]:
        """List function names that this handler can execute."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def setup(self) -> None:
        """Set up the runtime environment, including any necessary initializations."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def teardown(self) -> None:
        """Clean up the runtime environment, releasing any resources."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def execute(self, pfunc: PFunction, args: list[TensorLike]) -> list[TensorLike]:
        """Execute the provided PFunction with the given arguments."""
        raise NotImplementedError("Subclasses must implement this method.")
