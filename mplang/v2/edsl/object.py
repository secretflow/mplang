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

"""Object: Base class for runtime objects.

Base abstraction for distinguishing trace-time and interp-time execution.

- TraceObject: Defined in mplang.edsl.tracer
- InterpObject: Defined in mplang.edsl.interpreter
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mplang.v2.edsl.typing import BaseType

T = TypeVar("T", bound=BaseType)


class Object(ABC, Generic[T]):
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
    def type(self) -> T:
        """Type of the object (available in both trace and interp modes)."""

    # Note: Arithmetic operators (__add__, __mul__, etc.) are NOT defined here.
    # They should be provided by dialect-specific dispatch mechanisms since
    # different types (Tensor, Vector, SS) require different implementations.
