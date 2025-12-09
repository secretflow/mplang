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

from mplang.v2.dialects import store
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter


def _get_shard_uri(uri_base: str, rank: int) -> str:
    """Generate shard URI: {uri_base} (no suffix, relies on backend isolation)."""
    # Handle different schemes if necessary, for now assume simple path joining
    # or scheme preservation.
    if "://" in uri_base:
        scheme, _, path = uri_base.partition("://")
        # Ensure we don't double slash if path is absolute
        return f"{scheme}://{path}"
    else:
        # Default to fs:// for absolute paths (sandboxed)
        return f"fs://{uri_base}"


@store.save_p.def_impl
def save_impl(interpreter: Interpreter, op: Operation, obj_val: Any) -> Any:
    """Save implementation."""
    uri_base: str = op.attrs["uri_base"]

    # In SPMD, each interpreter instance corresponds to a party/rank.
    # We retrieve the rank from the interpreter context.
    # TODO: Standardize rank access in Interpreter base class.
    rank = getattr(interpreter, "rank", getattr(interpreter, "trace_pid", 0))

    # All ranks write data shard
    shard_uri = _get_shard_uri(uri_base, rank)

    # Use ObjectStore to put the value
    # Note: obj_val is the runtime value (e.g. TensorValue, TableValue, or raw)
    # We store it as is (pickle).
    interpreter.store.put(obj_val, uri=shard_uri)

    return obj_val


@store.load_p.def_impl
def load_impl(interpreter: Interpreter, op: Operation) -> Any:
    """Load implementation."""
    uri_base: str = op.attrs["uri_base"]
    # expected_type is in attrs but not needed for runtime loading (pickle handles it)

    rank = getattr(interpreter, "rank", getattr(interpreter, "trace_pid", 0))
    shard_uri = _get_shard_uri(uri_base, rank)

    return interpreter.store.get(shard_uri)
