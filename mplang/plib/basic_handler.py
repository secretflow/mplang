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

from mplang.core.base import TensorLike
from mplang.core.pfunc import PFunction, PFunctionHandler


class BasicHandler(PFunctionHandler):
    # override
    def setup(self) -> None:
        """Set up the runtime environment."""

    # override
    def teardown(self) -> None:
        """Clean up the runtime environment."""

    def list_fn_names(self) -> list[str]:
        """List function names that this handler can execute."""
        return ["Identity"]

    # override
    def execute(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        if pfunc.fn_type == "builtin.identity":
            if len(args) != 1:
                raise ValueError("Identity expects exactly one argument.")
            return args
        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
