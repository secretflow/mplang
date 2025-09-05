# Design Doc: Verifiable Computation Primitives

**Status:** Draft
**Owner:** [Your Name/Team]
**Last Updated:** September 5, 2025

## 1. Motivation & Goal

In many multi-party computation scenarios, a "full trust" model (where all parties trust each other) is not feasible, while a "zero trust" model relying purely on cryptographic techniques (like Multi-Party Computation or Homomorphic Encryption) can be prohibitively expensive in terms of performance.

This design introduces a **"partial trust" model** for MPLang, leveraging Trusted Execution Environments (TEEs). By designating one or more parties as trusted TEEs, we can offload sensitive computations to them. This approach offers a significant performance advantage over pure cryptographic methods, as computation inside the TEE can run on plaintext data, while still providing strong cryptographic guarantees of code integrity and data confidentiality to the other, untrusting parties.

Our goal is to build a frontend and backend that allows N data-providing parties to **cryptographically verify** that M TEE parties are genuine, are running the correct program, and are executing the exact same computation graph before any sensitive data is shared.

## 2. High-Level Flow

The attestation process is initiated by the `Driver` and transparently handled by the MPLang runtime, ensuring the user's script remains focused on business logic.

1. **Scripting & Compilation**: A user writes a Python script. Either the programmer explicitly, or a security-aware compiler automatically, inserts `quote_gen` and `quote_verify` calls at the appropriate points before sensitive data is sent to a TEE party. The final computation graph (MPIR) thus contains this security logic, which can be independently reviewed or formally verified. The `Driver` then compiles this into a final MPIR.

2. **Session Initiation (by Driver)**: Before execution, the `Driver` prepares a security context for the session:
    * It computes the hash of the MPIR: `mpir_hash`.
    * It generates a globally unique nonce for the session: `session_nonce`.
    * It signs the tuple `(mpir_hash, session_nonce)` with its private key to produce a `driver_signature`. This signature proves the authenticity and integrity of the computation to all parties.

3. **Distribution**: The `Driver` distributes the execution package to all participating parties. This package contains: `(MPIR, session_nonce, driver_signature)`.

4. **Runtime Verification**:
    * **Initial Check (All Parties)**: Upon receiving the package, every party's runtime first verifies the `driver_signature` against the `mpir_hash` and `session_nonce` using the `Driver`'s public key. This confirms the job's authenticity and prevents tampering.
    * **TEE Attestation (TEE Parties)**: When a TEE party's runtime encounters a `quote_gen` instruction, it automatically includes the `mpir_hash` and `session_nonce` from the execution context into the quote's `report_data`.
    * **Quote Verification (Data Parties)**: When a data-providing party's runtime **executes a `quote_verify` instruction,** it performs verification on the received quote. It independently reconstructs the expected `report_data` using the same `mpir_hash` and `session_nonce` and compares it against the data in the quote. If this check, or any other part of the quote verification, fails, the runtime **must immediately halt**.

5. **Secure Execution**: Once all verifications succeed, data-providing parties can securely transmit their data to the TEE parties, and the computation proceeds as defined in the MPIR.

### 3.1. Trust & Verification Logic

Our design relies on a chain of trust:

* **Trust in Driver**: All participating parties must trust the `Driver`. They have access to the `Driver`'s public key to verify its signature on the computation job.
* **Trust in TEE Hardware**: Data-providing parties trust the TEE hardware manufacturer (e.g., Intel, AMD) and have access to their public root certificates for quote verification.

The core logic is as follows:

* **The Provers:** The M TEE-based parties.
* **The Verifiers:** The N non-TEE data-providing parties.
* **The Proof's Content (Each TEE party proves to all data-providers that...)**:
    1. **"I am a real TEE."** (Authenticity): Proven by the quote's signature chaining back to a trusted hardware vendor.
    2. **"I am running the program you expect."** (Code Integrity): Proven by the program measurement (e.g., `MRENCLAVE`) in the quote.
    3. **"I am executing the specific, untampered job from our trusted Driver, for this specific session."** (Logic & Freshness): This is the most critical part. It is proven by the quote containing a commitment to `hash(mpir_hash, session_nonce)`, where both `mpir_hash` and `session_nonce` were provided by the trusted `Driver`.

## 4. TEE Frontend API

To enable this flow, we will introduce a new `mplang.frontend.tee` module. These functions are intended to be used within a `@mplang.function` traced graph. The compiler will then lower them into the appropriate backend instructions.

### 4.1. API Functions

The user-facing API is minimal. The complexity of nonce and hash management is handled by the framework.

| Function           | Description                                                                                                                                                                                          |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `quote_gen()`      | A core attestation function. When executed on a TEE party, it generates a quote. The `report_data` for the quote is **implicitly constructed** by the runtime from the session's `mpir_hash` and `session_nonce`. |
| `quote_verify(quote)`| Verifies a received `quote`. The expected `report_data` is also **implicitly constructed** by the runtime for comparison. Typically runs on non-TEE parties.                                        |

**Note on Data Transfer:** There are no explicit `send` or `receive` functions in this API. Data transfer (e.g., broadcasting the quote) is handled implicitly by the MPLang runtime when the computation graph indicates that data produced by one party is consumed by another. This is typically realized using the `pshfl` (Party Shuffle) primitive under the hood.

### 4.2. Conceptual Execution Flow

The following shows the logical sequence of operations. The `session_nonce` and `mpir_hash` are assumed to be present in the runtime context.

```python
# In a TEE Party's runtime (e.g., Party C):
# The runtime automatically accesses the hash and nonce from the job context.
report_data = hash(context.mpir_hash, context.session_nonce)
# The user's code simply calls quote_gen(), and the runtime provides the data.
c_quote, c_pubkey = tee.quote_gen(report_data=report_data) # report_data is implicit

# The quote and public key are broadcast to data-providers.

# In a Data-Provider's runtime (e.g., Party A):
# The runtime also automatically accesses the hash and nonce.
expected_report_data = hash(context.mpir_hash, context.session_nonce)
# The user's code calls quote_verify(c_quote), and the runtime provides
# the expected data for verification.
is_trusted = tee.quote_verify(c_quote, c_pubkey,
                              EXPECTED_C_PROGRAM_HASH,
                              expected_report_data=expected_report_data # implicit
                             )
assert(is_trusted)
```

### 4.3. Example MPLang Implementation

With the simplified design, the user's code becomes much cleaner.

```python
import mplang
import mplang.simp as simp
import mplang.frontend.tee as tee # Proposed TEE frontend

# Parties are identified by an index.
# Example: 2 providers (A, B) and 1 TEE (C)
A, B, C = 0, 1, 2
DATA_PROVIDERS = [A, B]
TEE_PARTIES = [C]

@mplang.function
def secure_computation_with_attestation(a_data, b_data):
    # The mpir_hash and session_nonce are handled by the Driver and Runtime.
    # The user's logic does not need to be aware of them.

    # 1. Each TEE party generates a quote.
    # The runtime implicitly provides the correct report_data.
    quotes_and_keys = [simp.runAt(p, tee.quote_gen) for p in TEE_PARTIES]

    # 2. Each data provider verifies the quotes from all TEE parties.
    # The runtime implicitly provides the expected report_data for verification.
    for p in DATA_PROVIDERS:
        is_trusted = simp.runAt(p, tee.quote_verify, quotes_and_keys)
        assert is_trusted

    # 3. Proceed with actual computation...
    # ...
    return ...
```

## 5. Security Analysis: Attack Vectors & Defenses

This design is intended to counter the following specific attacks:

### 5.1. Attack: Malicious `C` Impersonation

* **Vector:** An attacker runs a non-TEE program that pretends to be the trusted party `C`.
* **Defense:** `quote_verify` will fail at the very first step. The attacker cannot produce a `quote` with a valid signature that chains back to a TEE hardware vendor.

### 5.2. Attack: Wrong Program Execution

* **Vector:** A genuine TEE `C` is running the wrong version of the program (e.g., an old, vulnerable version).
* **Defense:** `quote_verify` will fail. The `program_hash` inside the `quote` will not match the `EXPECTED_C_PROGRAM_HASH` that `A` and `B` expect.

### 5.3. Attack: Graph Mismatch (Logic Deception)

* **Vector:** The `Driver` (or a Man-in-the-Middle) sends a different **computation graph** to `C` than to `A` and `B`.
* **Defense:** `quote_verify` will fail. `C`'s `quote` will contain a commitment to `hash(evil_graph)`, while `A` and `B` will be expecting a commitment to `hash(good_graph)`. The verification of `report_data` will fail.

### 5.4. Attack: Replay Attack

* **Vector:** An attacker records a valid `quote` from a previous, legitimate session and replays it to trick data-providers into sending data.
* **Defense:** This is prevented by the `session_nonce`. The `Driver` generates a **fresh, unpredictable nonce for every single session**. The `quote` is cryptographically bound to this specific `session_nonce`. A replayed `quote` will contain an old, invalid nonce and will be rejected by `quote_verify`. This ensures the proof is "live" for the current session.

## 6. Implementation Guidance for Backend and Runtime

* The `quote_gen` function **MUST** be implemented by a dedicated TEE backend, which calls the underlying platform's TEE SDK (e.g., Intel SGX SDK's `sgx_create_quote`). The runtime is responsible for the IPC with the platform's quoting service daemon (e.g., `aesmd`).
* The `quote_verify` function requires access to a trust store containing the TEE vendor's public root certificates. This might be configured globally for the runtime.
* The runtime **MUST** treat the `assert(is_trusted)` step as a critical security boundary. A failed assertion must result in an immediate and clean termination of the computation for that party. No further operations should be possible.
