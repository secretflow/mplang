# Design Doc: Verifiable Computation Primitives

**Status:** Draft
**Owner:** [Your Name/Team]
**Last Updated:** September 5, 2025

## 1. Motivation & Goal

In many multi-party computation scenarios, a "full trust" model (where all parties trust each other) is not feasible, while a "zero trust" model relying purely on cryptographic techniques (like Multi-Party Computation or Homomorphic Encryption) can be prohibitively expensive in terms of performance.

This design introduces a **"partial trust" model** for MPLang, leveraging Trusted Execution Environments (TEEs). By designating one or more parties as trusted TEEs, we can offload sensitive computations to them. This approach offers a significant performance advantage over pure cryptographic methods, as computation inside the TEE can run on plaintext data, while still providing strong cryptographic guarantees of code integrity and data confidentiality to the other, untrusting parties.

Our goal is to build a frontend and backend that allows N data-providing parties to **cryptographically verify** that M TEE parties are genuine, are running the correct program, and are executing the exact same computation graph before any sensitive data is shared.

## 2. High-Level Flow

The attestation and verification process is deeply integrated into the MPLang compiler and runtime lifecycle. It applies to any setup with N data-providing parties (verifiers) and M TEE-enabled parties (provers).

The flow is as follows:

1.  **Scripting**: A user writes a Python script defining the computation logic, treating TEE parties as regular participants.

2.  **Tracing & Compilation**: As MPLang traces the script into a computation graph (MPIR), it identifies data flows that cross trust boundaries (e.g., plaintext from a data-provider to a TEE party). Based on the cluster's security layout, the compiler **automatically inserts** the necessary attestation and data-sealing nodes (`quote_gen`, `quote_verify`, and data transfer operations like `pshfl`) into the graph.

3.  **Distribution**: The final, augmented computation graph (MPIR) is serialized and distributed by the `Driver` to all N+M participating parties.

4.  **Runtime Verification**: Each party's runtime executes the graph.
    *   When a TEE party encounters a `quote_gen` instruction, it calls its local TEE backend to produce a cryptographic quote.
    *   When a data-providing party encounters a `quote_verify` instruction, it verifies the received quote. If the verification fails for any reason (invalid signature, wrong program measurement, mismatched graph hash), the runtime **must immediately halt**.

5.  **Secure Execution**: Once verification succeeds, the data-providing parties can securely transmit their data to the TEE parties (e.g., by encrypting it with a public key obtained from the TEE's quote). The rest of the computation then proceeds, executed by MPLang's various backends (e.g., SPU for MPC, DuckDB for local SQL, or the TEE backend for trusted computation).

## 3. Trust & Verification Logic

### 3.1. Pre-existing Trust Assumptions

Our design relies on the following foundational trust relationships:

* **Trust in Business Logic:** All participants (`A`, `B`, `C`) trust the **computation graph** they receive from the `Driver`. The mechanism for this trust (e.g., `Driver` digitally signing the graph) is **out of scope** for this protocol but is a prerequisite. All parties start with a shared, trusted **computation graph**.
* **Trust in TEE Hardware:** `A` and `B` trust the TEE hardware manufacturer (e.g., Intel, AMD) and have access to their public root certificates. This is a fundamental platform assumption.

### 3.2. Who Proves What to Whom?

This is the core logic of the protocol, generalized for any number of parties:

* **The Provers:** The M TEE-based parties.
* **The Verifiers:** The N non-TEE data-providing parties.
* **The Proof's Content (Each TEE party proves to all data-providers that...)**:
    1. **"I am a real TEE."** (Authenticity): This is proven by a cryptographic signature on the `quote` that chains back to the TEE hardware manufacturer's trusted root certificate.
    2. **"I am running the program you expect."** (Code Integrity): The `quote` contains a measurement of the TEE party's program binary (e.g., `MRENCLAVE`). The verifiers will check this against a known-good value.
    3. **"I am about to execute the exact same computation graph you have, right now."** (Logic & Freshness): This is the most critical part. The `quote` must also contain a commitment to the specific **computation graph** for this session, combined with a freshness guarantee (via nonces).

The combination of these three proofs establishes a secure context for the computation.

## 4. TEE Frontend API

To enable this flow, we will introduce a new `mplang.frontend.tee` module. These functions are intended to be used within a `@mplang.function` traced graph. The compiler will then lower them into the appropriate backend instructions.

### 4.1. API Functions

| Function           | Description                                                                                                                                                                                          |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gen_nonce()`      | Generates a fresh, cryptographic random number on the executing party.                                                                                                                               |
| `quote_gen(data)`  | A core attestation function. It requests the TEE platform to generate a `quote` that cryptographically binds the machine's identity, the program's identity, and the input `data`. **Must run in a TEE.** |
| `quote_verify(...)`| Verifies a received `quote` against a set of expectations (program hash, data hash, nonces). Typically runs on non-TEE parties.                                                                    |

**Note on Data Transfer:** There are no explicit `send` or `receive` functions in this API. Data transfer between parties (e.g., sending nonces to the TEE party or broadcasting the quote) is handled implicitly by the MPLang runtime when the computation graph indicates that data produced by one party is consumed by another. This is typically realized using the `pshfl` (Party Shuffle) primitive under the hood.

### 4.2. Conceptual Execution Flow

The following shows the logical sequence of operations for a single data-provider `A` and a single TEE party `C`. This logic extends to N providers and M TEEs.

```python
# In Party A's script (a data-provider)
# The computation graph hash is known/provided.
graph_hash = hash(computation_graph)
my_nonce = tee.gen_nonce()

# The nonce is sent to C implicitly by being used in a function call on C.
# C waits to receive all nonces from all N data-providers.

# C generates the quote and a public key for data sealing.
# report_data = hash(graph_hash, nonce_from_A, nonce_from_B, ...)
# c_quote, c_pubkey = tee.quote_gen(report_data) # Executed on C

# The quote and public key are broadcast from C to all data-providers.
# Party A receives them and verifies.
is_trusted = tee.quote_verify(c_quote, c_pubkey,
                              EXPECTED_C_PROGRAM_HASH,
                              graph_hash,
                              my_nonce,
                              # ... and all other nonces
                             )
assert(is_trusted) # Runtime MUST halt here if false

# If trusted, A can now send its sensitive data to C.
encrypted_data = encrypt(c_pubkey, my_sensitive_data)
# `encrypted_data` is sent to C implicitly.
```

### 4.3. Example MPLang Implementation

Here is how the flow could be implemented in an MPLang script.

```python
import mplang
import mplang.simp as simp
import mplang.frontend.tee as tee # Proposed TEE frontend

# Parties are identified by an index. Let's say N providers (0..N-1) and M TEEs (N..N+M-1).
# Example: 2 providers (A, B) and 1 TEE (C)
A, B, C = 0, 1, 2
DATA_PROVIDERS = [A, B]
TEE_PARTIES = [C]

@mplang.function
def secure_computation_with_attestation(a_data, b_data):
    # The traced graph of this function is the "computation_graph"
    graph_hash = "..." # Assume this can be obtained or is a constant

    # 1. All data providers generate a nonce.
    nonces = [simp.runAt(p, tee.gen_nonce) for p in DATA_PROVIDERS]

    # 2. Each TEE party generates a quote.
    # It implicitly gathers the graph_hash and all nonces.
    # (Broadcasting nonces from all providers to all TEEs happens here).
    quotes_and_keys = [simp.runAt(p, tee.quote_gen, graph_hash, nonces) for p in TEE_PARTIES]

    # 3. Each data provider verifies the quotes from all TEE parties.
    # (Broadcasting quotes from all TEEs to all providers happens here).
    for p in DATA_PROVIDERS:
        # Each provider `p` needs all original nonces to perform verification.
        is_trusted = simp.runAt(p, tee.quote_verify, quotes_and_keys, graph_hash, nonces)
        assert is_trusted

    # 4. Proceed with actual computation...
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

* **Vector:** An attacker records a valid `quote` from a previous, legitimate session and replays it to `A` and `B` to trick them into sending data.
* **Defense:** This is what the nonces are for. `A` and `B` generate **fresh, unpredictable nonces** for *every single session*. The `quote` is cryptographically bound to these specific nonces. A replayed `quote` will contain old nonces and will be rejected by `quote_verify`. This ensures the proof is "live".

## 6. Implementation Guidance for Backend and Runtime

* The `quote_gen` function **MUST** be implemented by a dedicated TEE backend, which calls the underlying platform's TEE SDK (e.g., Intel SGX SDK's `sgx_create_quote`). The runtime is responsible for the IPC with the platform's quoting service daemon (e.g., `aesmd`).
* The `quote_verify` function requires access to a trust store containing the TEE vendor's public root certificates. This might be configured globally for the runtime.
* The runtime **MUST** treat the `assert(is_trusted)` step as a critical security boundary. A failed assertion must result in an immediate and clean termination of the computation for that party. No further operations should be possible.
