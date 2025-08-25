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

from abc import ABC, abstractmethod
from typing import Any

from jax.tree_util import PyTreeDef

from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction


class FEOp(ABC):
    """Base class for all frontend operations.

    This class provides a common interface for all frontend operations
    that can be compiled and executed by the mplang system.
    """

    @abstractmethod
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Compile the function with given arguments.

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: The compiled function,
            input arguments, and output tree definition.
        """
