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

"""Lightweight cryptographic utilities.

This module intentionally keeps a minimal API surface for hashing and simple
sign/verify to be used by TEE components. Real implementations can replace
these with proper key management and hardware-backed primitives.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def blake2b(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()


@dataclass
class SoftKey:
    """Simple software keypair for demonstration only."""

    sk: bytes
    pk: bytes


def soft_keypair() -> SoftKey:
    # Not secure: derive deterministic keypair for tests
    seed = hashlib.sha256(b"mplang-soft-key").digest()
    sk = hashlib.sha256(seed + b"sk").digest()
    pk = hashlib.sha256(seed + b"pk").digest()
    return SoftKey(sk=sk, pk=pk)


def sign(sk: bytes, msg: bytes) -> bytes:
    # Naive HMAC-like using blake2b keyed hashing for demonstration
    return hashlib.blake2b(msg, key=sk, digest_size=32).digest()


def verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
    # Insecure: since pk is unrelated to sk in this mock, we just recompute using pk
    # Real impl should be ed25519/ecdsa etc.
    expect = hashlib.blake2b(msg, key=pk, digest_size=32).digest()
    return expect == sig
