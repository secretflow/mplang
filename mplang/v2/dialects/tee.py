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

"""TEE (Trusted Execution Environment) dialect for mplang.v2 EDSL.

This dialect provides primitives for TEE remote attestation, enabling secure
computation where:
1. Data providers can verify they're communicating with genuine TEE hardware
2. The TEE proves it's running the expected code (measurement)
3. Session keys are established via attested key exchange

Architecture:
    PublicKey (from crypto.kem_keygen)
        ↓ quote_gen(pk)
    Quote  (cryptographic attestation proof)
        ↓ (transfer to verifier)
        ↓ attest(quote)
    AttestedKey[curve]  (verified TEE public key)
        ↓ crypto.kem_derive(local_sk, attested_pk)
    SharedSecret  (secure channel with TEE)

Supported Platforms:
    - "mock": Insecure mock for local testing (default)
    - "sgx": Intel SGX (DCAP attestation)
    - "tdx": Intel TDX
    - "sev": AMD SEV-SNP

Example:
```python
from mplang.v2.dialects import tee, crypto

# On TEE side: generate keypair and quote
sk, pk = crypto.kem_keygen("x25519")
quote = tee.quote_gen(pk)  # Bind public key in attestation

# On verifier side: verify quote and get attested key
attested_pk = tee.attest(quote)

# Establish secure channel
verifier_sk, verifier_pk = crypto.kem_keygen("x25519")
shared_secret = crypto.kem_derive(verifier_sk, attested_pk)
```

Security Note:
    The mock implementation is NOT SECURE and should only be used for
    development and testing. Production deployments must use real TEE
    backends (SGX, TDX, SEV).
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects.crypto import PublicKeyType
from mplang.v2.edsl import serde

# ==============================================================================
# --- Type Definitions
# ==============================================================================

KeyCurve = Literal["x25519", "secp256k1"]


@serde.register_class
class QuoteType(elt.BaseType):
    """Type for a TEE attestation quote.

    A quote is a cryptographic proof that:
    1. Code is running in a genuine TEE (signed by hardware)
    2. The TEE is running expected code (measurement/MRENCLAVE)
    3. The quote binds a specific ephemeral public key

    The quote can be verified by anyone with access to the TEE vendor's
    root certificates (Intel, AMD, etc.).
    """

    def __str__(self) -> str:
        return "TEEQuote"

    def __repr__(self) -> str:
        return "QuoteType()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuoteType)

    def __hash__(self) -> int:
        return hash("QuoteType")

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "tee.QuoteType"

    def to_json(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> QuoteType:
        return cls()


@serde.register_class
class AttestedKeyType(elt.BaseType):
    """Type for an attested public key extracted from a verified quote.

    This represents a public key that has been cryptographically proven to
    belong to a genuine TEE running specific code. It can be used for:
    - Key exchange (KEM/ECDH) to establish secure channels
    - Encryption directly to the TEE

    The key is trusted because:
    1. The quote signature chains to a trusted TEE vendor root
    2. The measurement in the quote matches expected code
    3. The public key is bound in the quote's report_data

    Attributes:
        curve: Cryptographic curve of the key
    """

    def __init__(self, curve: str = "x25519"):
        self.curve: KeyCurve = curve  # type: ignore[assignment]

    def __str__(self) -> str:
        return f"AttestedKey[{self.curve}]"

    def __repr__(self) -> str:
        return f"AttestedKeyType(curve={self.curve!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AttestedKeyType):
            return False
        return self.curve == other.curve

    def __hash__(self) -> int:
        return hash(("AttestedKeyType", self.curve))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "tee.AttestedKeyType"

    def to_json(self) -> dict[str, Any]:
        return {"curve": self.curve}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AttestedKeyType:
        return cls(curve=data["curve"])


@serde.register_class
class MeasurementType(elt.BaseType):
    """Type for TEE code measurement (e.g., MRENCLAVE, MRTD).

    Represents a cryptographic hash of the code and initial configuration
    running inside the TEE. Used to verify the TEE is running expected code.
    """

    def __str__(self) -> str:
        return "TEEMeasurement"

    def __repr__(self) -> str:
        return "MeasurementType()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MeasurementType)

    def __hash__(self) -> int:
        return hash("MeasurementType")

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "tee.MeasurementType"

    def to_json(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MeasurementType:
        return cls()


# ==============================================================================
# --- Primitives
# ==============================================================================

quote_gen_p = el.Primitive[el.Object]("tee.quote_gen")
attest_p = el.Primitive[el.Object]("tee.attest")
get_measurement_p = el.Primitive[el.Object]("tee.get_measurement")


# ==============================================================================
# --- Abstract Evaluation (Type Inference)
# ==============================================================================


@quote_gen_p.def_abstract_eval
def _quote_gen_ae(
    pk: elt.BaseType,
) -> QuoteType:
    """Generate a TEE quote binding the provided public key.

    Args:
        pk: Public key to bind in the quote (must be PublicKeyType from crypto.kem_keygen)

    Returns:
        QuoteType representing the attestation proof

    Raises:
        TypeError: If pk is not a PublicKeyType
    """
    if not isinstance(pk, PublicKeyType):
        raise TypeError(
            f"quote_gen expects PublicKeyType (from crypto.kem_keygen), "
            f"got {type(pk).__name__}"
        )
    return QuoteType()


@attest_p.def_abstract_eval
def _attest_ae(
    quote: QuoteType,
    *,
    expected_curve: KeyCurve = "x25519",
) -> AttestedKeyType:
    """Verify a quote and extract the attested public key.

    Args:
        quote: The TEE quote to verify
        expected_curve: Expected curve of the bound key

    Returns:
        AttestedKeyType containing the verified public key
    """
    return AttestedKeyType(curve=expected_curve)


@get_measurement_p.def_abstract_eval
def _get_measurement_ae(
    quote: QuoteType,
) -> MeasurementType:
    """Extract the measurement (MRENCLAVE/MRTD) from a quote.

    Args:
        quote: The TEE quote

    Returns:
        MeasurementType containing the code measurement
    """
    return MeasurementType()


# ==============================================================================
# --- User-facing API (Helper Functions)
# ==============================================================================


def quote_gen(
    pk: el.Object,
) -> el.Object:
    """Generate a TEE attestation quote binding the provided public key.

    This operation runs inside the TEE and produces a quote that:
    1. Proves the code is running in genuine TEE hardware
    2. Contains the measurement (hash) of the running code
    3. Binds the provided public key in the report_data

    Args:
        pk: Public key to bind (typically from crypto.kem_keygen)

    Returns:
        Object[QuoteType] - The attestation quote

    Example:
        >>> sk, pk = crypto.kem_keygen("x25519")
        >>> quote = tee.quote_gen(pk)  # Bind pk in attestation
    """
    return quote_gen_p.bind(pk)


def attest(
    quote: el.Object,
    expected_curve: KeyCurve = "x25519",
) -> el.Object:
    """Verify a TEE quote and extract the attested public key.

    This operation runs on the verifier side and:
    1. Validates the quote signature against TEE vendor roots
    2. Checks the measurement matches expected values
    3. Extracts the bound public key

    After verification, the returned key is trusted to belong to a genuine
    TEE running the expected code.

    Args:
        quote: The TEE quote to verify
        expected_curve: Expected curve of the bound key

    Returns:
        Object[AttestedKeyType] - The verified public key

    Raises:
        AttestationError: If verification fails

    Example:
        >>> attested_pk = tee.attest(quote)
        >>> # Now safe to derive shared secret with TEE
        >>> shared = crypto.kem_derive(my_sk, attested_pk)
    """
    return attest_p.bind(quote, expected_curve=expected_curve)


def get_measurement(quote: el.Object) -> el.Object:
    """Extract the code measurement from a quote.

    The measurement is a cryptographic hash of the code running in the TEE.
    For SGX this is MRENCLAVE, for TDX this is MRTD, etc.

    Args:
        quote: The TEE quote

    Returns:
        Object[MeasurementType] - The code measurement
    """
    return get_measurement_p.bind(quote)


__all__ = [
    "AttestedKeyType",
    "KeyCurve",
    "MeasurementType",
    "QuoteType",
    "attest",
    "attest_p",
    "get_measurement",
    "get_measurement_p",
    "quote_gen",
    "quote_gen_p",
]
