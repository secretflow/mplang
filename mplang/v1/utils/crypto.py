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

Currently only exposes `blake2b` which is used by mock crypto backends.
Previously contained mock signing/key utilities which were unused in code
paths and have been removed to avoid confusion.
"""

from __future__ import annotations

import hashlib


def blake2b(data: bytes) -> bytes:
    """Return 32-byte BLAKE2b digest for the given data."""
    return hashlib.blake2b(data, digest_size=32).digest()


__all__ = ["blake2b"]
