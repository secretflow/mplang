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

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for typing to avoid import cycles at runtime.
    from mplang.v1.core.mpobject import MPContext

# The global working context.
_g_ctx: MPContext | None = None


def cur_ctx() -> MPContext:
    if _g_ctx is None:
        # Keep the original error text for backward compatibility with callers/tests.
        raise ValueError("Interpreter not set. Please call set_interp() first.")
    return _g_ctx


def set_ctx(ctx: MPContext) -> None:
    global _g_ctx
    _g_ctx = ctx


@contextlib.contextmanager
def with_ctx(tmp_ctx: MPContext) -> Iterator[MPContext]:
    global _g_ctx
    saved = _g_ctx  # Directly save the global interpreter reference
    try:
        _g_ctx = tmp_ctx
        yield tmp_ctx
    finally:
        # Restore the previous interpreter even if it was None
        _g_ctx = saved
