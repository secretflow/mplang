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

"""Mock TEE backend implementation for local testing.

WARNING: This implementation is NOT SECURE and should only be used for
development and testing purposes. It simulates TEE attestation without
any actual hardware security guarantees.

For production deployments, use real TEE backends that integrate with
hardware attestation (Intel SGX DCAP, AMD SEV-SNP, etc.).
"""

from __future__ import annotations

import base64
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from mplang.v2.backends.crypto_impl import BytesValue, PublicKeyValue
from mplang.v2.dialects import tee
from mplang.v2.edsl import serde
from mplang.v2.runtime.value import Value

if TYPE_CHECKING:
    from mplang.v2.edsl.graph import Operation
    from mplang.v2.runtime.interpreter import Interpreter


# ==============================================================================
# --- Mock Data Structures
# ==============================================================================


@serde.register_class
@dataclass
class MockQuoteValue(Value):
    """Mock TEE quote structure.

    In production, this would contain:
    - Platform-specific attestation data
    - Signature from TEE hardware
    - Measurement (MRENCLAVE/MRTD)
    - User-provided report_data (bound public key hash)

    For mock purposes, we store the bound public key directly.
    The quote is the only mock-specific structure needed because it represents
    hardware-generated attestation data that doesn't exist outside a real TEE.
    """

    _serde_kind: ClassVar[str] = "tee_impl.MockQuoteValue"

    platform: str
    bound_pk: bytes  # The public key bytes bound in this quote (32 bytes for x25519)
    suite: str  # The KEM suite (e.g., "x25519")

    def to_json(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "bound_pk": base64.b64encode(self.bound_pk).decode("ascii"),
            "suite": self.suite,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MockQuoteValue:
        return cls(
            platform=data["platform"],
            bound_pk=base64.b64decode(data["bound_pk"]),
            suite=data["suite"],
        )

    def to_bytes(self) -> bytes:
        """Serialize quote for transmission."""
        # Format: [platform_len:1][platform][suite_len:1][suite][pk]
        platform_bytes = self.platform.encode("utf-8")
        suite_bytes = self.suite.encode("utf-8")
        return (
            bytes([len(platform_bytes)])
            + platform_bytes
            + bytes([len(suite_bytes)])
            + suite_bytes
            + self.bound_pk
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> MockQuoteValue:
        """Deserialize quote."""
        platform_len = data[0]
        platform = data[1 : 1 + platform_len].decode("utf-8")
        suite_start = 1 + platform_len
        suite_len = data[suite_start]
        suite = data[suite_start + 1 : suite_start + 1 + suite_len].decode("utf-8")
        pk_start = suite_start + 1 + suite_len
        bound_pk = data[pk_start : pk_start + 32]
        return cls(platform=platform, bound_pk=bound_pk, suite=suite)


# ==============================================================================
# --- Implementation Functions
# ==============================================================================


def _emit_mock_warning(operation: str) -> None:
    """Emit a warning that mock TEE is being used."""
    warnings.warn(
        f"Insecure mock TEE operation '{operation}' in use. "
        "NOT secure; for local testing only.",
        UserWarning,
        stacklevel=4,
    )


@tee.quote_gen_p.def_impl
def _quote_gen_impl(
    interpreter: Interpreter,
    op: Operation,
    pk: PublicKeyValue,
) -> MockQuoteValue:
    """Generate a mock TEE quote binding the provided public key.

    In a real TEE, this would:
    1. Hash the public key into report_data
    2. Generate a hardware-signed attestation report
    3. Package everything into a quote structure

    For mock, we just wrap the public key in a MockQuoteValue.

    Args:
        interpreter: The interpreter context
        op: The operation being executed
        pk: Public key to bind (from crypto.kem_keygen)

    Returns:
        MockQuoteValue containing the bound public key
    """
    _emit_mock_warning("tee.quote_gen")

    if not isinstance(pk, PublicKeyValue):
        raise TypeError(
            f"quote_gen expects PublicKeyValue from crypto.kem_keygen, "
            f"got {type(pk).__name__}"
        )

    # In a real implementation, the platform would be detected from the environment
    platform = "mock"

    return MockQuoteValue(
        platform=platform,
        bound_pk=pk.key_bytes,
        suite=pk.suite,
    )


@tee.attest_p.def_impl
def _attest_impl(
    interpreter: Interpreter,
    op: Operation,
    quote: MockQuoteValue | BytesValue,
) -> PublicKeyValue:
    """Verify a mock quote and extract the attested public key.

    In a real implementation, this would:
    1. Verify the quote signature against TEE vendor root certificates
    2. Check the measurement matches expected code hash
    3. Extract and return the verified public key

    For mock, we just extract the public key directly (no real verification).

    Args:
        interpreter: The interpreter context
        op: The operation being executed
        quote: The quote to verify (MockQuoteValue or BytesValue for serialized quotes)

    Returns:
        PublicKeyValue - the verified public key, ready for use with kem_derive
    """
    _emit_mock_warning("tee.attest")

    # Handle different quote formats
    if isinstance(quote, MockQuoteValue):
        mock_quote = quote
    elif isinstance(quote, BytesValue):
        mock_quote = MockQuoteValue.from_bytes(quote.unwrap())
    elif isinstance(quote, (bytes, bytearray, np.ndarray)):
        # Fallback for raw bytes (backwards compatibility / real TEE interop)
        if isinstance(quote, np.ndarray):
            quote = bytes(quote)
        mock_quote = MockQuoteValue.from_bytes(bytes(quote))
    else:
        raise TypeError(f"Expected MockQuoteValue or BytesValue, got {type(quote)}")

    # Return a real PublicKeyValue that can be used directly with kem_derive
    return PublicKeyValue(
        suite=mock_quote.suite,
        key_bytes=mock_quote.bound_pk,
    )


__all__ = [
    "MockQuoteValue",
]
