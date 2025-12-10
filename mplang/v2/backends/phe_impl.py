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

"""PHE Runtime Implementation using LightPHE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from lightphe import LightPHE
from lightphe.models.Ciphertext import Ciphertext

from mplang.v2.dialects import phe
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter


class PHEContext:
    """Wraps LightPHE context."""

    def __init__(self, algorithm_name: str = "Paillier", key_size: int = 2048):
        # Normalize algorithm name (LightPHE expects capitalized names)
        normalized_name = algorithm_name.capitalize()
        self.cs = LightPHE(algorithm_name=normalized_name, key_size=key_size)

    def encrypt(self, value: int) -> Ciphertext:
        return self.cs.encrypt(value)

    def decrypt(self, ct: Ciphertext) -> int:
        return cast(int, self.cs.decrypt(ct))


@dataclass
class PHEEncoder:
    """Simple fixed-point encoder."""

    scale: float


@dataclass
class WrappedCiphertext:
    ct: Ciphertext
    ctx: PHEContext

    def __add__(self, other: Any) -> WrappedCiphertext:
        if isinstance(other, WrappedCiphertext):
            # ct + ct
            new_ct = self.ct + other.ct
            return WrappedCiphertext(new_ct, self.ctx)
        elif isinstance(other, int):
            # ct + int -> ct + encrypt(int)
            ct_other = self.ctx.encrypt(other)
            new_ct = self.ct + ct_other
            return WrappedCiphertext(new_ct, self.ctx)
        return NotImplemented

    def __mul__(self, other: Any) -> WrappedCiphertext:
        if isinstance(other, int):
            # ct * int
            new_ct = self.ct * other
            return WrappedCiphertext(new_ct, self.ctx)
        return NotImplemented


@phe.keygen_p.def_impl
def keygen_impl(
    interpreter: Interpreter, op: Operation, *args: Any
) -> tuple[PHEContext, PHEContext]:
    key_size = op.attrs.get("key_size", 2048)
    scheme = op.attrs.get("scheme", "Paillier")

    ctx = PHEContext(algorithm_name=scheme, key_size=key_size)

    return ctx, ctx


@phe.create_encoder_p.def_impl
def create_encoder_impl(
    interpreter: Interpreter, op: Operation, *args: Any
) -> PHEEncoder:
    fxp_bits = op.attrs.get("fxp_bits", 16)
    scale = 2.0**fxp_bits
    return PHEEncoder(scale=scale)


@phe.encode_p.def_impl
def encode_impl(
    interpreter: Interpreter, op: Operation, value: float, encoder: PHEEncoder
) -> int:
    return int(value * encoder.scale)


@phe.decode_p.def_impl
def decode_impl(
    interpreter: Interpreter, op: Operation, value: int, encoder: PHEEncoder
) -> float:
    return float(value) / encoder.scale


@phe.encrypt_p.def_impl
def encrypt_impl(
    interpreter: Interpreter, op: Operation, value: int, pk: PHEContext
) -> WrappedCiphertext:
    ct = pk.encrypt(value)
    return WrappedCiphertext(ct, pk)


@phe.decrypt_p.def_impl
def decrypt_impl(
    interpreter: Interpreter, op: Operation, wct: WrappedCiphertext, sk: PHEContext
) -> int:
    return sk.decrypt(wct.ct)


@phe.add_cc_p.def_impl
def add_cc_impl(
    interpreter: Interpreter,
    op: Operation,
    lhs: WrappedCiphertext,
    rhs: WrappedCiphertext,
) -> WrappedCiphertext:
    return lhs + rhs


@phe.add_cp_p.def_impl
def add_cp_impl(
    interpreter: Interpreter, op: Operation, lhs: WrappedCiphertext, rhs: int
) -> WrappedCiphertext:
    return lhs + rhs


@phe.mul_cp_p.def_impl
def mul_cp_impl(
    interpreter: Interpreter, op: Operation, lhs: WrappedCiphertext, rhs: int
) -> WrappedCiphertext:
    return lhs * rhs
