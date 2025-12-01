from typing import Any

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects.tensor import _ElementwiseTracer, elementwise, run_jax


def add(x: Any, y: Any) -> Any:
    """Element-wise addition."""
    ctx = el.get_current_context()
    if isinstance(ctx, _ElementwiseTracer):
        return run_jax(lambda a, b: a + b, x, y)
    if (
        isinstance(x, el.TraceObject)
        and isinstance(x.type, elt.ScalarType)
        and isinstance(y, el.TraceObject)
        and isinstance(y.type, elt.ScalarType)
    ):
        return run_jax(lambda a, b: a + b, x, y)
    return elementwise(lambda a, b: a + b, x, y)


def sub(x: Any, y: Any) -> Any:
    """Element-wise subtraction."""
    ctx = el.get_current_context()
    if isinstance(ctx, _ElementwiseTracer):
        return run_jax(lambda a, b: a - b, x, y)
    if (
        isinstance(x, el.TraceObject)
        and isinstance(x.type, elt.ScalarType)
        and isinstance(y, el.TraceObject)
        and isinstance(y.type, elt.ScalarType)
    ):
        return run_jax(lambda a, b: a - b, x, y)
    return elementwise(lambda a, b: a - b, x, y)


def mul(x: Any, y: Any) -> Any:
    """Element-wise multiplication."""
    ctx = el.get_current_context()
    if isinstance(ctx, _ElementwiseTracer):
        return run_jax(lambda a, b: a * b, x, y)
    if (
        isinstance(x, el.TraceObject)
        and isinstance(x.type, elt.ScalarType)
        and isinstance(y, el.TraceObject)
        and isinstance(y.type, elt.ScalarType)
    ):
        return run_jax(lambda a, b: a * b, x, y)
    return elementwise(lambda a, b: a * b, x, y)


def div(x: Any, y: Any) -> Any:
    """Element-wise division."""
    ctx = el.get_current_context()
    if isinstance(ctx, _ElementwiseTracer):
        return run_jax(lambda a, b: a / b, x, y)
    if (
        isinstance(x, el.TraceObject)
        and isinstance(x.type, elt.ScalarType)
        and isinstance(y, el.TraceObject)
        and isinstance(y.type, elt.ScalarType)
    ):
        return run_jax(lambda a, b: a / b, x, y)
    return elementwise(lambda a, b: a / b, x, y)


def neg(x: Any) -> Any:
    """Element-wise negation."""
    ctx = el.get_current_context()
    if isinstance(ctx, _ElementwiseTracer):
        return run_jax(lambda a: -a, x)
    if isinstance(x, el.TraceObject) and isinstance(x.type, elt.ScalarType):
        return run_jax(lambda a: -a, x)
    return elementwise(lambda a: -a, x)


def patch_object_operators() -> None:
    """Patch Object class to support arithmetic operators via tensor primitives."""
    from mplang.v2.edsl.object import Object

    def _add(self: Object, other: Any) -> Any:
        return add(self, other)

    def _radd(self: Object, other: Any) -> Any:
        return add(other, self)

    def _sub(self: Object, other: Any) -> Any:
        return sub(self, other)

    def _rsub(self: Object, other: Any) -> Any:
        return sub(other, self)

    def _mul(self: Object, other: Any) -> Any:
        return mul(self, other)

    def _rmul(self: Object, other: Any) -> Any:
        return mul(other, self)

    def _truediv(self: Object, other: Any) -> Any:
        return div(self, other)

    def _rtruediv(self: Object, other: Any) -> Any:
        return div(other, self)

    def _neg(self: Object) -> Any:
        return neg(self)

    Object.__add__ = _add  # type: ignore
    Object.__radd__ = _radd  # type: ignore
    Object.__sub__ = _sub  # type: ignore
    Object.__rsub__ = _rsub  # type: ignore
    Object.__mul__ = _mul  # type: ignore
    Object.__rmul__ = _rmul  # type: ignore
    Object.__truediv__ = _truediv  # type: ignore
    Object.__rtruediv__ = _rtruediv  # type: ignore
    Object.__neg__ = _neg  # type: ignore
