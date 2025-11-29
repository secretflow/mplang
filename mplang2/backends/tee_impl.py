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

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from mplang2.backends.crypto_impl import RuntimePublicKey
from mplang2.dialects import tee

if TYPE_CHECKING:
    from mplang2.edsl.graph import Operation
    from mplang2.edsl.interpreter import Interpreter


# ==============================================================================
# --- Mock Data Structures
# ==============================================================================


@dataclass
class MockQuote:
    """Mock TEE quote structure.

    In production, this would contain:
    - Platform-specific attestation data
    - Signature from TEE hardware
    - Measurement (MRENCLAVE/MRTD)
    - User-provided report_data (bound public key hash)

    For mock purposes, we just store the bound public key directly.
    """

    platform: str
    bound_pk: np.ndarray  # The public key bound in this quote
    measurement: np.ndarray  # Mock measurement (32 bytes)

    def to_bytes(self) -> bytes:
        """Serialize quote for transmission."""
        # Simple format: [platform_len:1][platform][pk:32][measurement:32]
        platform_bytes = self.platform.encode("utf-8")
        header = bytes([len(platform_bytes)])
        return (
            header
            + platform_bytes
            + self.bound_pk.tobytes()
            + self.measurement.tobytes()
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> MockQuote:
        """Deserialize quote."""
        platform_len = data[0]
        platform = data[1 : 1 + platform_len].decode("utf-8")
        pk_start = 1 + platform_len
        pk = np.frombuffer(data[pk_start : pk_start + 32], dtype=np.uint8).copy()
        measurement = np.frombuffer(
            data[pk_start + 32 : pk_start + 64], dtype=np.uint8
        ).copy()
        return cls(platform=platform, bound_pk=pk, measurement=measurement)


@dataclass
class MockAttestedKey:
    """Mock attested key structure.

    Represents a public key that has been "verified" (in mock mode)
    to belong to a TEE. In production, this would only be produced
    after successful quote verification.
    """

    platform: str
    curve: str
    public_key: np.ndarray  # The verified public key bytes


@dataclass
class MockMeasurement:
    """Mock measurement structure.

    Represents the code measurement (MRENCLAVE, MRTD, etc.) extracted
    from a quote.
    """

    platform: str
    hash_bytes: np.ndarray  # 32-byte measurement hash


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


def _get_mock_measurement() -> np.ndarray:
    """Generate a mock measurement (would be MRENCLAVE/MRTD in production)."""
    # In production, this would be the actual enclave measurement
    # For mock, we use a fixed pattern to make testing deterministic
    return np.array(
        [0xDE, 0xAD, 0xBE, 0xEF] * 8,
        dtype=np.uint8,
    )


@tee.quote_gen_p.def_impl
def _quote_gen_impl(
    interpreter: Interpreter,
    op: Operation,
    pk: RuntimePublicKey,
) -> MockQuote:
    """Generate a mock TEE quote binding the provided public key.

    Args:
        interpreter: The interpreter context
        op: The operation being executed
        pk: Public key to bind (from crypto.kem_keygen)

    Returns:
        MockQuote containing the bound public key
    """
    _emit_mock_warning("tee.quote_gen")

    if not isinstance(pk, RuntimePublicKey):
        raise TypeError(
            f"quote_gen expects RuntimePublicKey from crypto.kem_keygen, "
            f"got {type(pk).__name__}"
        )

    platform = op.attrs.get("platform", "mock")
    pk_arr = np.frombuffer(pk.key_bytes, dtype=np.uint8)[:32].copy()

    # Pad to 32 bytes if needed
    if len(pk_arr) < 32:
        pk_arr = np.pad(pk_arr, (0, 32 - len(pk_arr)), mode="constant")

    measurement = _get_mock_measurement()

    return MockQuote(
        platform=platform,
        bound_pk=pk_arr,
        measurement=measurement,
    )


@tee.attest_p.def_impl
def _attest_impl(
    interpreter: Interpreter,
    op: Operation,
    quote: Any,
) -> MockAttestedKey:
    """Verify a mock quote and extract the attested public key.

    Args:
        interpreter: The interpreter context
        op: The operation being executed
        quote: The quote to verify (MockQuote or bytes)

    Returns:
        MockAttestedKey containing the verified public key
    """
    _emit_mock_warning("tee.attest")

    # Get expected curve from attributes
    expected_curve = op.attrs.get("expected_curve", "x25519")

    # Handle different quote formats
    if isinstance(quote, MockQuote):
        mock_quote = quote
    elif isinstance(quote, (bytes, bytearray)):
        mock_quote = MockQuote.from_bytes(bytes(quote))
    else:
        raise TypeError(f"Expected MockQuote or bytes, got {type(quote)}")

    # In production, we would:
    # 1. Verify quote signature against TEE vendor root
    # 2. Check measurement against expected value
    # 3. Verify report_data contains H(pk)
    #
    # For mock, we just extract the public key directly

    return MockAttestedKey(
        platform=mock_quote.platform,
        curve=expected_curve,
        public_key=mock_quote.bound_pk,
    )


@tee.get_measurement_p.def_impl
def _get_measurement_impl(
    interpreter: Interpreter,
    op: Operation,
    quote: Any,
) -> MockMeasurement:
    """Extract the measurement from a quote.

    Args:
        interpreter: The interpreter context
        op: The operation being executed
        quote: The quote to extract measurement from

    Returns:
        MockMeasurement containing the code measurement
    """
    _emit_mock_warning("tee.get_measurement")

    # Handle different quote formats
    if isinstance(quote, MockQuote):
        mock_quote = quote
    elif isinstance(quote, (bytes, bytearray)):
        mock_quote = MockQuote.from_bytes(bytes(quote))
    else:
        raise TypeError(f"Expected MockQuote or bytes, got {type(quote)}")

    return MockMeasurement(
        platform=mock_quote.platform,
        hash_bytes=mock_quote.measurement,
    )


# ==============================================================================
# --- Conversion Helpers
# ==============================================================================


def attested_key_to_bytes(attested_key: MockAttestedKey) -> jnp.ndarray:
    """Convert an attested key to bytes for use with crypto operations.

    This allows AttestedKey to be used with crypto.kem_derive.
    """
    return jnp.array(attested_key.public_key, dtype=jnp.uint8)


def measurement_to_bytes(measurement: MockMeasurement) -> jnp.ndarray:
    """Convert a measurement to bytes for comparison or hashing."""
    return jnp.array(measurement.hash_bytes, dtype=jnp.uint8)


__all__ = [
    "MockAttestedKey",
    "MockMeasurement",
    "MockQuote",
    "attested_key_to_bytes",
    "measurement_to_bytes",
]
