"""Object: Base class for runtime objects.

Base abstraction for distinguishing trace-time and interp-time execution.

- TraceObject: Defined in mplang.edsl.tracer
- InterpObject: Defined in mplang.edsl.interpreter
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mplang.edsl.typing import BaseType


class Object(ABC):
    """Base class for MPLang runtime objects.

    This is a Driver-side abstraction used for:
    1. Distinguishing between trace-time and interp-time objects
    2. Providing uniform operation interfaces (arithmetic, attribute access, etc.)
    3. Enabling polymorphic handling by the Tracer

    Subclasses:
    - TraceObject: Trace-time object (holds a Value in Graph IR) - in mplang.edsl.tracer
    - InterpObject: Interp-time object (holds backend-specific runtime data) - in mplang.edsl.interpreter
    """

    @property
    @abstractmethod
    def type(self) -> BaseType:
        """Type of the object (available in both trace and interp modes)."""

    # Note: Arithmetic operators (__add__, __mul__, etc.) are NOT defined here.
    # They should be provided by dialect-specific dispatch mechanisms since
    # different types (Tensor, SIMD_HE, SS) require different implementations.
