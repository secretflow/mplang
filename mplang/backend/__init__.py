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

"""Backend public API.

Exports:
        RuntimeContext: per-rank backend execution context (explicit op->kernel binding).
        bind_all_ops: idempotent bootstrap establishing default op bindings.
"""

from .context import RuntimeContext, bind_all_ops  # noqa: F401
