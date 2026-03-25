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

"""Store Runtime Implementation."""

from __future__ import annotations

from typing import Any

from mplang.dialects import store
from mplang.edsl.graph import Operation
from mplang.runtime.interpreter import Interpreter


@store.save_p.def_impl
def save_impl(interpreter: Interpreter, op: Operation, obj_val: Any) -> Any:
    """Save implementation — delegates to ObjectStore.put(uri=...)."""
    uri_base: str = op.attrs["uri_base"]
    if interpreter.store is None:
        raise RuntimeError("Interpreter has no ObjectStore configured. Cannot save.")
    interpreter.store.put(obj_val, uri=uri_base)
    return obj_val


@store.load_p.def_impl
def load_impl(interpreter: Interpreter, op: Operation) -> Any:
    """Load implementation — delegates to ObjectStore.get()."""
    uri_base: str = op.attrs["uri_base"]
    if interpreter.store is None:
        raise RuntimeError("Interpreter has no ObjectStore configured. Cannot load.")
    return interpreter.store.get(interpreter.store.resolve_uri(uri_base))
