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

"""Primitive: User-facing API for building atomic operations.

Provides the Primitive class for defining operations that automatically work in
both trace mode (record to Graph IR) and interp mode (execute immediately).

See Primitive class documentation for detailed usage examples.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from jax.tree_util import tree_map

from mplang.v2.edsl.context import get_current_context, get_default_context
from mplang.v2.edsl.object import Object

if TYPE_CHECKING:
    from mplang.v2.edsl.typing import BaseType

T_Ret = TypeVar("T_Ret")


class Primitive(Generic[T_Ret]):
    """Atomic operation definition (similar to JAX Primitive).

    A Primitive represents an atomic operation that can be:
    1. **Traced**: Records operation to Graph IR (via abstract_eval or trace)
    2. **Executed**: Runs via backend execution of Graph IR

    Attributes:
        name: Unique name of the primitive (e.g., "add", "mul", "encrypt")
        _abstract_eval: Type inference function (type → type)
        _trace: Custom trace logic for complex operations

    Example:
        >>> # Define custom FHE encryption primitive
        >>> encrypt_p = Primitive("fhe_encrypt")
        >>>
        >>> @encrypt_p.def_abstract_eval
        >>> def encrypt_abstract(x_type):
        >>>     from mplang.v2.edsl.typing import Vector
        >>>     return Vector[x_type.dtype, x_type.shape]
        >>>
        >>> # Execution happens via Graph IR → Backend
        >>> # Backend handles FHE library calls based on operation type
        >>>
        >>> # Usage
        >>> plaintext = TraceObject(...)
        >>> ciphertext = encrypt_p.bind(plaintext)  # Records to Graph IR
    """

    def __init__(self, name: str):
        """Initialize a primitive with a unique name.

        Args:
            name: Unique identifier for this primitive (e.g., "add", "encrypt")
        """
        self.name = name
        self._abstract_eval: Callable[..., BaseType | Sequence[BaseType]] | None = None
        self._trace: Callable[..., Any] | None = None
        self._impl: Callable[..., Any] | None = None

    def def_impl(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Define execution logic for this primitive in the interpreter.

        This function is called by the Interpreter during eager execution or
        when evaluating a graph.

        Args:
            fn: Function that implements the operation.
                Signature: (interpreter, op, *args) -> result

        Returns:
            The same function (for decorator pattern)
        """
        self._impl = fn
        # Register with the global interpreter registry
        from mplang.v2.edsl.registry import register_impl

        register_impl(self.name, fn)
        return fn

    def def_abstract_eval(
        self, fn: Callable[..., BaseType | Sequence[BaseType]]
    ) -> Callable[..., BaseType | Sequence[BaseType]]:
        """Define type inference rule for this primitive.

        This function is called during tracing to infer output types from input types.
        Supports both single-output and multi-output primitives.

        Supported signatures:
        1. Positional form (variable number of input types):
           (*in_types: BaseType, **attrs) -> BaseType | Sequence[BaseType]

        2. Flat form (input types as list):
           (in_types: list[BaseType], **attrs) -> BaseType | Sequence[BaseType]

        Args:
            fn: Function that takes input types and returns output type(s)

        Returns:
            The same function (for decorator pattern)

        Example (positional form):
            >>> add_p = Primitive("add")
            >>>
            >>> @add_p.def_abstract_eval
            >>> def add_abstract(x_type: BaseType, y_type: BaseType) -> BaseType:
            >>>     assert x_type == y_type, "Inputs must have same type"
            >>>     return x_type

        Example (positional form, multi-output):
            >>> split_p = Primitive("split")
            >>>
            >>> @split_p.def_abstract_eval
            >>> def split_abstract(x_type: BaseType, *, num_splits: int) -> list[BaseType]:
            >>>     return [x_type] * num_splits

        Example (flat form):
            >>> concat_p = Primitive("concat")
            >>>
            >>> @concat_p.def_abstract_eval
            >>> def concat_abstract(in_types: list[BaseType], *, axis: int = 0) -> BaseType:
            >>> # Variable number of inputs
            >>>     return in_types[0]  # Concatenated type
        """
        self._abstract_eval = fn
        return fn

    def def_trace(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Define custom trace logic for this primitive.

        This method enables full control over the tracing process, suitable for
        complex scenarios like:
        - Integrating external functions (JAX, FHE, etc.)
        - Accepting arbitrary PyTree inputs mixing Objects and constants
        - Producing arbitrary PyTree outputs

        The decorated function receives raw args/kwargs and returns the result PyTree.
        The tracer automatically handles:
        - Extracting Objects from input PyTree (via var_morph)
        - Recording morph structure to Operation attrs
        - Flattening output PyTree
        - Reconstructing output structure during interpretation

        Signature: (*args, **kwargs) -> Object | PyTree[Object]

        Args:
            fn: Custom trace function that takes arbitrary args/kwargs and
                returns result PyTree (can contain Objects and constants)

        Returns:
            The same function (for decorator pattern)

        Example (JAX integration):
            >>> run_jax_p = Primitive("run_jax")
            >>>
            >>> @run_jax_p.def_trace
            >>> def run_jax_trace(jax_fn: Callable, *args, **kwargs):
            >>> # args/kwargs can mix Objects and constants
            >>> # Compile JAX function and execute
            >>>     result = compile_and_run(jax_fn, args, kwargs)
            >>>     return result  # Can be any PyTree structure
            >>>
            >>> # Example (multi-output):
            >>> split_p = Primitive("split")
            >>>
            >>> @split_p.def_trace
            >>> def split_trace(x: Object, *, num_splits: int):
            >>> # Call underlying operations
            >>>     parts = [slice_p.bind(x, i) for i in range(num_splits)]
            >>>     return parts  # Returns list of Objects
        """
        self._trace = fn
        return fn

    def bind(self, *args: Any, **kwargs: Any) -> T_Ret:
        """Bind arguments and execute/trace the primitive.

        This is the main user-facing API. It automatically chooses between:
        - **Trace mode**: Record operation to Graph IR (if in Tracer context)
        - **Interp mode**: Execute Graph IR via backend (if in Interpreter context)

        Behavior depends on which method was used to define the primitive:
        - **def_abstract_eval**: Positional args must be Objects (inputs),
          kwargs must be plain values (attrs). Returns single Object or list[Object].
        - **def_trace**: Both args and kwargs can mix Objects and plain values.
          Returns arbitrary PyTree structure.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Object | PyTree[Object] - Result structure depends on primitive definition

        Raises:
            RuntimeError: If neither abstract_eval nor trace is defined
            TypeError: If using def_abstract_eval and kwargs contain Object instances

        Example:
            >>> # With def_abstract_eval (simple form)
            >>> z = add_p.bind(x, y)  # x, y are Objects
            >>>
            >>> # With def_trace (full form)
            >>> result = run_jax_p.bind(fn, obj1, 42, obj2, k=3.14)
            >>> # Mixing Objects (obj1, obj2) and constants (42, 3.14)
        """
        # Get current context
        ctx = get_current_context()
        if ctx is None:
            ctx = get_default_context()

        def lift_if_object(x: Any) -> Any:  # Add type annotation
            return ctx.lift(x) if isinstance(x, Object) else x

        lifted_args, lifted_kwargs = tree_map(lift_if_object, (args, kwargs))

        # Execute in context
        return cast(T_Ret, ctx.bind_primitive(self, lifted_args, lifted_kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> T_Ret:
        """Syntactic sugar for bind(): primitive(*args, **kwargs) == primitive.bind(*args, **kwargs)."""
        return self.bind(*args, **kwargs)


# ============================================================================
# Decorator: @primitive for defining primitives in a concise way
# ============================================================================


def primitive(name: str) -> Callable[[Callable], Primitive]:
    """Decorator for defining primitives in a concise way.

    This is a convenience decorator that creates a Primitive and registers
    the decorated function as its abstract_eval rule.

    Args:
        name: Unique name for the primitive

    Returns:
        Decorator function

    Example:
        >>> @primitive("my_custom_op")
        >>> def my_op_abstract(x_type: BaseType, y_type: BaseType) -> BaseType:
        >>> # Type inference logic
        >>>     return x_type
        >>>
        >>> # The decorator returns a Primitive instance
        >>> my_op_p = my_op_abstract
        >>>
        >>> # Use it (execution via Graph IR → Backend)
        >>> z = my_op_p.bind(x, y)
    """

    def decorator(fn: Callable) -> Primitive[Any]:
        p: Primitive[Any] = Primitive(name)
        p.def_abstract_eval(fn)
        return p

    return decorator


__all__ = [
    "Primitive",
    "primitive",
]
