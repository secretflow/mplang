# Design Doc: Verifiable Computation Primitives (TEE v2 API)

## 1. Motivation & Goal

In many MPC scenarios, a full-trust model is unrealistic, while a zero-trust
model relying purely on cryptography (MPC/HE) can be too slow for practical
use. Trusted Execution Environments (TEEs) enable a pragmatic partial-trust
model: we let selected parties execute on plaintext inside hardware-enforced
boundaries while maintaining cryptographic assurance of code integrity and data
confidentiality to untrusted parties.

Goal: Provide a frontend and backend such that N data providers can, before
sharing sensitive data, cryptographically verify that M TEE parties are genuine,
run the expected environment, and execute the exact same MPIR.

## 2. High-Level Flow

Attestation is initiated by the Driver and handled by the runtime, keeping user
code focused on business logic.

1. Scripting & Compilation: Before data is sent to TEE, insert `tee.quote()`
   and `tee.attest(quote)` into the program. In mock mode we may also use
   `crypto.keygen` for test keys; in production `tee.quote()` provides an
   ephemeral public key and `tee.attest` derives a symmetric session key via
   KEM/ECDH. The final MPIR contains auditable security logic. The Driver
   compiles and distributes the session security context.

2. Session Initiation (Driver):
   - Compute `program_hash` (hash of MPIR)
   - Specify `runtime_measurement` (e.g., SGX MRENCLAVE)
   - Generate `session_nonce`
   - Sign `(program_hash, runtime_measurement, session_nonce)` as
     `driver_signature`

3. Distribution: Send `(MPIR, program_hash, runtime_measurement, session_nonce,
   driver_signature)` to all parties.

4. Runtime Verification:
   - Initial Check (All): Verify `driver_signature` over the tuple
   - TEE Attestation (TEE parties): execute `tee.quote()`; the TEE generates an
     ephemeral keypair inside the enclave and emits a quote that binds
     `report_data = H(program_hash || session_nonce || H(ephemeral_pubkey))`
   - Quote Verification & KEM (Data parties): execute `tee.attest(quote)`;
     verify vendor chain, measurement, and `report_data`. On success, perform
     KEM/ECDH encapsulation against the attested `ephemeral_pubkey` to derive a
     symmetric `session_key` and produce a `kem_ct` header to be delivered to the
     TEE. The data party then uses `session_key` to encrypt sealed payloads and
     attaches `kem_ct` (once per session or per sender) so the TEE can
     decapsulate and derive the same key.

5. Secure Execution: After verification, data parties encrypt and send their
   data to TEE and computation proceeds as defined by MPIR.

## 3. Trust & Verification

- Trust in Driver: parties have the Driver public key to verify job signature
- Trust in TEE Hardware: parties trust vendor root certs for quote verification
- Proof content:
  - Authentic TEE (vendor chain)
  - Expected runtime (measurement)
  - Same program (bind `program_hash` + `session_nonce` + `H(ephemeral_pubkey)`)

## 4. Frontend API (TEE + Crypto)

These functions are used inside `@mplang.function`-traced graphs and lowered to
backend instructions. The API stays simple for users while allowing a mock mode
and a production-ready KEM/ECDH path under the hood.

- `mplang.frontend.crypto`
  - `keygen(length: int = 32) -> key` (mock/testing)
  - `enc(plaintext, key) -> ciphertext` (nonce-prefixed; AES-GCM recommended)
  - `dec(ciphertext, key) -> plaintext`

- `mplang.frontend.tee`
  - `quote() -> quote`
  - `attest(quote) -> session`

Notes:

- `tee.quote` runs on TEE; in production it generates an ephemeral keypair and
  binds the public key via `report_data = H(program_hash || session_nonce ||
  H(ephemeral_pubkey))` to respect platform limits (e.g., SGX 64B)
- `tee.attest` runs on data parties; on success it derives a symmetric
  `session_key` via KEM/ECDH and produces a `kem_ct` header to be delivered to
  the TEE so it can decapsulate the same `session_key`
- In mock mode (for local testing), `tee.quote(payload)` and `tee.attest(quote)`
  may still return a symmetric key directly; the high-level user semantics
  remain “encrypt-to-TEE before leaving the current security domain”
- Data transfer is implicit from the graph (typically via `pshfl`/`scatter`)

## 5. Sequence (conceptual)

```mermaid
sequenceDiagram
  participant Driver
  participant A as Data-Provider A
  participant B as Data-Provider B
  participant C as TEE Party C

  note right of Driver: 1) Session Init (Offline)
  Driver->>Driver: program_hash = hash(MPIR)
  Driver->>Driver: runtime_measurement = expected()
  Driver->>Driver: session_nonce = gen()
  Driver->>Driver: driver_signature = sign(tuple)

  note right of Driver: 2) Distribution & Runtime
  Driver->>A: (MPIR, program_hash, measurement, nonce, signature)
  Driver->>B: (MPIR, program_hash, measurement, nonce, signature)
  Driver->>C: (MPIR, program_hash, measurement, nonce, signature)

  note over A, C: Verify driver signature

  C->>C: generate ephemeral keypair inside TEE
  C->>C: report_data = H(program_hash, nonce, H(ephemeral_pubkey))
  C->>C: quote = tee.quote()

  C-->>A: quote
  C-->>B: quote

  A->>A: session = tee.attest(quote)
  B->>B: session = tee.attest(quote)

  note over A: session = { session_key, kem_ct }
  note over B: session = { session_key, kem_ct }

  A-->>C: send kem_ct (once) and encrypted data (nonce||ciphertext)
  B-->>C: send kem_ct (once) and encrypted data (nonce||ciphertext)

  note over A, C: Computation proceeds
```

## 6. Security Analysis

- Impersonation: non-TEE cannot produce a vendor-signed quote → verification fails
- Wrong Runtime: measurement mismatch vs Driver expectation → fail
- Wrong Program: `program_hash` or `H(ephemeral_pubkey)` mismatch in `report_data` → fail
- Graph Mismatch: Driver sends different MPIRs → mismatch in `report_data` → fail
- Replay: bound to fresh `session_nonce` → old quotes rejected
- Key exposure: the symmetric `session_key` is never embedded in the quote or
  any plaintext payload; it is derived via KEM/ECDH

## 7. Implementation Guidance

- Crypto frontend:
  - `enc/dec` (AES-GCM recommended; manage random nonce and bundle nonce with
    ciphertext)
  - When adopting production KEM, extend the sealed message format to carry
    `kem_ct` as a small header (see 7.1) without changing the user API

- TEE backend:
  - `quote()` integrates vendor SDK (e.g., SGX DCAP); generate an ephemeral
    keypair inside TEE and bind `report_data = H(program_hash || session_nonce ||
    H(ephemeral_pubkey))`
  - `attest(quote)` verifies chain, measurement, and `report_data`; performs
    KEM/ECDH encapsulation to derive `session_key` and produce `kem_ct`

- On any verification failure, the runtime must terminate the party/session

### 7.1. Sealed message format (production, proposed)

To carry KEM materials with minimal user API surface, sealed payloads SHOULD
include a small header so the TEE can decapsulate before decryption. A minimal
format is:

```text
| kem_ct_len (u32 LE) | kem_ct bytes | nonce (12B) | ciphertext ... |
```

This allows the same `enc/dec` user API while enabling production E2E security.
Implementations may cache `kem_ct` per session and omit it from subsequent
messages if a channel/session is already established.
