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

from mplang.core.dtype import UINT8
from mplang.core.tensor import TensorType
from mplang.frontend.base import stateless_mod

_TEE_MOD = stateless_mod("tee")


@_TEE_MOD.simple_op()
def quote(pk: TensorType) -> TensorType:
    """TEE quote generation binding the provided ephemeral public key.

    API (mock): quote(pk: u8[32]) -> (quote: u8[33])
    The mock encodes a 1-byte header + 32-byte pk.
    """
    _ = pk  # Mark as used for the decorator
    return TensorType(UINT8, (33,))


@_TEE_MOD.simple_op()
def attest(quote: TensorType) -> TensorType:
    """TEE quote verification returning the attested TEE public key.

    API (mock): attest(quote: u8[33]) -> tee_pk: u8[32]
    """
    _ = quote  # Mark as used for the decorator
    return TensorType(UINT8, (32,))
