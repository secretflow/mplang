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

This module provides the Primitive abstraction for defining atomic operations
in the MPLang EDSL. Similar to JAX's Primitive system, it separates:

1. **Abstract evaluation**: Type inference rules (trace-time)
2. **Implementation**: Concrete execution logic (interp-time)

Example:
    >>> # Define a new primitive
    >>> add_p = Primitive("add")
    >>>
    >>> @add_p.def_abstract_eval
    >>> def add_abstract(x_type, y_type):
    >>> # Type inference: both inputs must have same type
    >>>     assert x_type == y_type
    >>>     return x_type
    >>>
    >>> @add_p.def_impl
    >>> def add_impl(x, y):
    >>> # Implementation: actual execution logic
    >>>     return x.runtime_obj + y.runtime_obj
    >>>
    >>> # Use the primitive
    >>> z = add_p.bind(x, y)  # Traces or executes based on context
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mplang.edsl.object import InterpObject, Object, TraceObject
    from mplang.edsl.tracer import Tracer
    from mplang.edsl.typing import BaseType


class Primitive:
    """Atomic operation definition (similar to JAX Primitive).

    A Primitive represents an atomic operation that can be:
    1. **Traced**: Records operation to Graph IR (via abstract_eval)
    2. **Executed**: Runs immediately on runtime objects (via impl)

    Attributes:
        name: Unique name of the primitive (e.g., "add", "mul", "encrypt")
        _abstract_eval: Type inference function (type → type)
        _impl: Concrete implementation function (runtime_obj → runtime_obj)

    Example:
        >>> # Define custom FHE encryption primitive
        >>> encrypt_p = Primitive("fhe_encrypt")
        >>>
        >>> @encrypt_p.def_abstract_eval
        >>> def encrypt_abstract(x_type):
        >>>     from mplang.edsl.typing import SIMD_HE
        >>>     return SIMD_HE[x_type.dtype, x_type.shape]
        >>>
        >>> @encrypt_p.def_impl
        >>> def encrypt_impl(x):
        >>>     import tenseal as ts
        >>>     encrypted = ts.encrypt(x.runtime_obj)
        >>>     return InterpObject(encrypted, encrypt_abstract(x.type))
        >>>
        >>> # Usage
        >>> plaintext = InterpObject(jnp.array([1, 2, 3]), Tensor[f32, (3,)])
        >>> ciphertext = encrypt_p.bind(plaintext)
    """

    def __init__(self, name: str):
        """Initialize a primitive with a unique name.

        Args:
            name: Unique identifier for this primitive (e.g., "add", "encrypt")
        """
        self.name = name
        self._abstract_eval: Callable[..., BaseType] | None = None
        self._impl: Callable[..., Any] | None = None

    def def_abstract_eval(self, fn: Callable[..., BaseType]) -> Callable[..., BaseType]:
        """Define type inference rule for this primitive.

        This function is called during tracing to infer output types from input types.

        Args:
            fn: Function that takes input types and returns output type(s)

        Returns:
            The same function (for decorator pattern)

        Example:
            >>> add_p = Primitive("add")
            >>>
            >>> @add_p.def_abstract_eval
            >>> def add_abstract(x_type: BaseType, y_type: BaseType) -> BaseType:
            >>>     assert x_type == y_type, "Inputs must have same type"
            >>>     return x_type
        """
        self._abstract_eval = fn
        return fn

    def def_impl(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Define concrete implementation for this primitive.

        This function is called during interpretation to execute the operation
        on runtime objects.

        Args:
            fn: Function that takes InterpObjects and returns result

        Returns:
            The same function (for decorator pattern)

        Example:
            >>> add_p = Primitive("add")
            >>>
            >>> @add_p.def_impl
            >>> def add_impl(x: InterpObject, y: InterpObject) -> InterpObject:
            >>>     result = x.runtime_obj + y.runtime_obj
            >>>     return InterpObject(result, x.type)
        """
        self._impl = fn
        return fn

    def bind(self, *args: Object, **kwargs: Any) -> TraceObject | InterpObject:
        """Bind arguments and execute/trace the primitive.

        This is the main user-facing API. It automatically chooses between:
        - **Trace mode**: Record operation to Graph IR (if in Tracer context)
        - **Interp mode**: Execute immediately (if in Interpreter context or eager mode)

        Args:
            *args: Positional arguments (Object instances)
            **kwargs: Keyword arguments (plain Python values, not Objects)

        Returns:
            TraceObject (if tracing) or InterpObject (if interpreting)

        Raises:
            RuntimeError: If neither abstract_eval nor impl is defined
            TypeError: If kwargs contain Object instances (not allowed)

        Example:
            >>> # In trace mode
            >>> tracer = Tracer()
            >>> with tracer:
            >>>     z = add_p.bind(x, y)  # Returns TraceObject
            >>>
            >>> # In eager mode
            >>> z = add_p.bind(x, y)  # Returns InterpObject
        """
        from mplang.edsl.context import get_context

        # Validate kwargs: must not contain Objects
        for key, value in kwargs.items():
            from mplang.edsl.object import Object

            if isinstance(value, Object):
                raise TypeError(
                    f"Keyword argument '{key}' cannot be an Object. "
                    f"Only positional arguments can be Objects. "
                    f"Use plain Python values (int, float, str, etc.) for kwargs."
                )

        # Get current context
        ctx = get_context()
        current_tracer = ctx.current_tracer

        if current_tracer is not None:
            # Trace mode: Record to Graph IR
            return self._bind_trace(current_tracer, args, kwargs)
        else:
            # Interp mode: Execute immediately
            return self._bind_interp(args, kwargs)

    def _bind_trace(
        self, tracer: Tracer, args: tuple[Object, ...], kwargs: dict[str, Any]
    ) -> TraceObject:
        """Bind in trace mode (record to Graph IR).

        Args:
            tracer: Current Tracer context
            args: Positional arguments (Objects)
            kwargs: Keyword arguments (plain values)

        Returns:
            TraceObject wrapping the result Value

        Raises:
            RuntimeError: If abstract_eval is not defined
        """
        if self._abstract_eval is None:
            raise RuntimeError(
                f"Primitive '{self.name}' has no abstract_eval rule. "
                f"Define it using @{self.name}_p.def_abstract_eval"
            )

        from mplang.edsl.object import TraceObject

        # Promote InterpObjects to TraceObjects if needed
        trace_args = []
        for arg in args:
            if isinstance(arg, TraceObject):
                trace_args.append(arg)
            else:
                # InterpObject → TraceObject (promote to graph)
                trace_args.append(tracer.promote(arg))

        # Get input types
        input_types = [arg.type for arg in trace_args]

        # Infer output type using abstract_eval
        output_type = self._abstract_eval(*input_types, **kwargs)

        # Add operation to graph using tracer.graph.add_op()
        input_values = [arg._graph_value for arg in trace_args]

        # Use Graph.add_op() which handles Value creation and Operation registration
        result_value = tracer.graph.add_op(
            opcode=self.name,
            inputs=input_values,
            output_types=[output_type],
            attrs=kwargs,
        )

        # add_op returns Value or list[Value], we know it's Value for single output
        if isinstance(result_value, list):
            result_value = result_value[0]

        # Return TraceObject wrapping the result Value
        return TraceObject(result_value, tracer)

    def _bind_interp(
        self, args: tuple[Object, ...], kwargs: dict[str, Any]
    ) -> InterpObject:
        """Bind in interp mode (execute immediately).

        Args:
            args: Positional arguments (Objects)
            kwargs: Keyword arguments (plain values)

        Returns:
            InterpObject wrapping the result runtime object

        Raises:
            RuntimeError: If impl is not defined
        """
        if self._impl is None:
            raise RuntimeError(
                f"Primitive '{self.name}' has no implementation. "
                f"Define it using @{self.name}_p.def_impl"
            )

        # Execute implementation
        result = self._impl(*args, **kwargs)
        return result


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
        >>> # Define implementation separately
        >>> @my_op_p.def_impl
        >>> def my_op_impl(x, y):
        >>>     return x.runtime_obj + y.runtime_obj
        >>>
        >>> # Use it
        >>> z = my_op_p.bind(x, y)
    """

    def decorator(fn: Callable) -> Primitive:
        p = Primitive(name)
        p.def_abstract_eval(fn)
        return p

    return decorator


# ============================================================================
# Pre-defined Primitives (common operations)
# ============================================================================

# Arithmetic primitives
add_p = Primitive("add")
mul_p = Primitive("mul")
sub_p = Primitive("sub")
div_p = Primitive("div")

# TODO: Define abstract_eval and impl for these primitives
# This will be done in a follow-up PR along with kernel integration

__all__ = [
    "Primitive",
    # Pre-defined primitives
    "add_p",
    "div_p",
    "mul_p",
    "primitive",
    "sub_p",
]
