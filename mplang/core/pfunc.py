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

from mplang.core.mptype import TensorLike, TensorType


class PFunction:
    """A Party Function represents a computation unit that can be executed by a single party.

    PFunction serves as a unified interface for describing single-party computations
    in multi-party computing scenarios. It can represent both:
    1. Built-in operations (e.g., "spu.makeshares", "builtin.read")
    2. User-defined programmable functions with custom code

    The PFunction accepts a list of TensorLike inputs and produces TensorLike outputs,
    and can be:
    - Expressed and defined in the mplang frontend
    - Serialized for transmission between components
    - Interpreted and executed by backend runtime engines

    Args:
        fn_type: The type/category identifier of this PFunction, indicating which
            backend or handler should process it (e.g., "spu.makeshares", "builtin.read",
            "mlir.stablehlo"). This serves as a routing mechanism for execution.
        ins_info: Tensor type information for input parameters
        outs_info: Tensor type information for output values
        fn_name: Optional name of the function. For programmable functions, this is
            the user-defined function name. For built-in operations, this may be
            None or a descriptive identifier.
        fn_text: Optional serialized function body. For programmable functions, this
            contains the actual code (e.g., MLIR, bytecode, source code). For built-in
            operations, this is typically None.
        **kwargs: Additional attributes and metadata specific to the function type.
            These are used to pass execution parameters, configuration, and context
            information to the backend handlers.
    """

    # Required fields - these define the core execution context
    fn_type: str  # Unique identifier for backend routing
    ins_info: tuple[TensorType, ...]
    outs_info: tuple[TensorType, ...]

    # Optional fields for programmable functions
    fn_name: str | None  # Function name (for programmable functions)
    fn_text: str | None  # Function body/code (for programmable functions)

    # Custom attributes and metadata
    attrs: MappingProxyType[str, Any]  # Execution parameters and metadata

    def __init__(
        self,
        fn_type: str,
        ins_info: Sequence[TensorType],
        outs_info: Sequence[TensorType],
        *,
        fn_name: str | None = None,
        fn_text: str | None = None,
        **kwargs: Any,
    ):
        self.fn_type = fn_type
        self.fn_name = fn_name
        self.fn_text = fn_text
        self.ins_info = tuple(ins_info)
        self.outs_info = tuple(outs_info)
        # Make attrs immutable to ensure PFunction immutability
        # Create a copy first, then wrap it in MappingProxyType
        self.attrs = MappingProxyType(copy.copy(kwargs))

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
