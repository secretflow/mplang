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
"""OpaqueType: placeholder spec for non-tensor/table backend kernel arguments.

Runtime validation currently only recognizes TensorType and TableType for structural
checks. Any other spec object in `ins_info` / `outs_info` is passed through without
shape/dtype enforcement. OpaqueType formalizes this usage and provides a readable
repr for debugging and IR dumps.

Typical use (e.g. crypto / phe keys):
    PFunction(
        fn_type="phe.encrypt",
        ins_info=(TensorType.from_obj(pt), OpaqueType("public_key")),
        outs_info=(OpaqueType("ciphertext"),),
        ...
    )

This allows the second argument to bypass tensor validation while still documenting
its semantic role.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OpaqueType"]


@dataclass(frozen=True)
class OpaqueType:
    name: str = "opaque"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Opaque<{self.name}>"
