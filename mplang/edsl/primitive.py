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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mplang.edsl.object import Object
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

    def bind(self, *args: Object, **kwargs: Any) -> Object:
        """Bind arguments and execute/trace the primitive.

        This is the main user-facing API. It automatically chooses between:
        - **Trace mode**: Record operation to Graph IR (if in Tracer context)
        - **Interp mode**: Execute immediately (if in Interpreter context or eager mode)

        Args:
            *args: Positional arguments (Object instances)
            **kwargs: Keyword arguments (plain Python values, not Objects)

        Returns:
            Object (TraceObject if tracing, InterpObject if interpreting)

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
        from mplang.edsl.context import get_current_context, get_default_interpreter

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
        ctx = get_current_context()

        if ctx is not None:
            # Use current context (Tracer or Interpreter)
            return ctx.bind_primitive(self, args, kwargs)
        else:
            # No context: use default interpreter (eager mode)
            return get_default_interpreter().bind_primitive(self, args, kwargs)


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


# Define abstract_eval for arithmetic primitives
@add_p.def_abstract_eval
def _add_abstract(x_type: BaseType, y_type: BaseType) -> BaseType:
    """Type inference for addition: returns the type of the first operand."""
    # TODO: Add proper type checking and unification
    return x_type


@mul_p.def_abstract_eval
def _mul_abstract(x_type: BaseType, y_type: BaseType) -> BaseType:
    """Type inference for multiplication: returns the type of the first operand."""
    # TODO: Add proper type checking and unification
    return x_type


@sub_p.def_abstract_eval
def _sub_abstract(x_type: BaseType, y_type: BaseType) -> BaseType:
    """Type inference for subtraction: returns the type of the first operand."""
    # TODO: Add proper type checking and unification
    return x_type


@div_p.def_abstract_eval
def _div_abstract(x_type: BaseType, y_type: BaseType) -> BaseType:
    """Type inference for division: returns the type of the first operand."""
    # TODO: Add proper type checking and unification
    return x_type


# Define impl for arithmetic primitives (eager execution)
@add_p.def_impl
def _add_impl(x: Object, y: Object) -> Object:
    """Eager execution of addition."""
    from mplang.edsl.interpreter import InterpObject

    if not isinstance(x, InterpObject) or not isinstance(y, InterpObject):
        raise TypeError("add_p.impl expects InterpObject operands")

    # TODO: Dispatch to appropriate backend executor based on type
    # For now, simple addition (assumes runtime_obj supports +)
    result_data = x.runtime_obj + y.runtime_obj
    return InterpObject(result_data, x.type)


@mul_p.def_impl
def _mul_impl(x: Object, y: Object) -> Object:
    """Eager execution of multiplication."""
    from mplang.edsl.interpreter import InterpObject

    if not isinstance(x, InterpObject) or not isinstance(y, InterpObject):
        raise TypeError("mul_p.impl expects InterpObject operands")

    result_data = x.runtime_obj * y.runtime_obj
    return InterpObject(result_data, x.type)


@sub_p.def_impl
def _sub_impl(x: Object, y: Object) -> Object:
    """Eager execution of subtraction."""
    from mplang.edsl.interpreter import InterpObject

    if not isinstance(x, InterpObject) or not isinstance(y, InterpObject):
        raise TypeError("sub_p.impl expects InterpObject operands")

    result_data = x.runtime_obj - y.runtime_obj
    return InterpObject(result_data, x.type)


@div_p.def_impl
def _div_impl(x: Object, y: Object) -> Object:
    """Eager execution of division."""
    from mplang.edsl.interpreter import InterpObject

    if not isinstance(x, InterpObject) or not isinstance(y, InterpObject):
        raise TypeError("div_p.impl expects InterpObject operands")

    result_data = x.runtime_obj / y.runtime_obj
    return InterpObject(result_data, x.type)


__all__ = [
    "Primitive",
    # Pre-defined primitives
    "add_p",
    "div_p",
    "mul_p",
    "primitive",
    "sub_p",
]
