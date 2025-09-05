# Design Doc: Verifiable Computation Primitives

**Status:** Draft
**Owner:** [Your Name/Team]
**Last Updated:** September 5, 2025

## 1. Motivation & Goal

Our Multi-Party Language (MPLang) needs to support scenarios where some parties (`A`, `B`) don't trust each other but are willing to trust a third party `C` running in a TEE (Trusted Execution Environment).

The goal of this design is to introduce a minimal set of frontend functions and a corresponding backend protocol that allows `A` and `B` to **cryptographically verify** that `C` is a genuine TEE instance, running the **correct program**, and executing the **exact same computation graph** they have, before they send any sensitive data to it.

This document outlines the "how" and "why" to guide the implementation of the frontend and backend.

## 2. High-Level Flow

The entire process is orchestrated by a `Driver` which distributes the same computation logic to all parties. The flow is as follows:

1. **Distribution:** The `Driver` script defines a computation, which is traced into a **computation graph** and sent to all participants (`A`, `B`, and `C`).
2. **Challenge & Attestation:** `A` and `B` challenge `C` to prove its identity and its commitment to the received **computation graph**. `C` responds with a cryptographic proof (`quote`).
3. **Verification:** `A` and `B` verify this proof.
4. **Execution:** If verification succeeds, all parties proceed with the secure computation defined in the graph.

This is a **blocking** process. If verification fails for any party, the entire computation must halt.

## 3. Trust & Verification Logic

### 3.1. Pre-existing Trust Assumptions

Our design relies on the following foundational trust relationships:

* **Trust in Business Logic:** All participants (`A`, `B`, `C`) trust the **computation graph** they receive from the `Driver`. The mechanism for this trust (e.g., `Driver` digitally signing the graph) is **out of scope** for this protocol but is a prerequisite. All parties start with a shared, trusted **computation graph**.
* **Trust in TEE Hardware:** `A` and `B` trust the TEE hardware manufacturer (e.g., Intel, AMD) and have access to their public root certificates. This is a fundamental platform assumption.

### 3.2. Who Proves What to Whom?

This is the core logic of the protocol:

* **The Prover:** The TEE-based party `C`.
* **The Verifiers:** The non-TEE parties `A` and `B`.
* **The Proof's Content (`C` proves to `A` and `B` that...)**:
    1. **"I am a real TEE."** (Authenticity): This is proven by a cryptographic signature on the `quote` that chains back to the TEE hardware manufacturer's trusted root certificate.
    2. **"I am running the program you expect."** (Code Integrity): The `quote` contains a measurement of `C`'s program binary (e.g., `MRENCLAVE`). `A` and `B` will check this against a known-good value.
    3. **"I am about to execute the exact same computation graph you have, right now."** (Logic & Freshness): This is the most critical part. The `quote` must also contain a commitment to the specific **computation graph** for this session, combined with a freshness guarantee.

The combination of these three proofs establishes a secure context for the computation.

## 4. TEE Frontend API & Execution Flow

To implement this flow, we will introduce a new `mplang.frontend.tee` module. This module will provide high-level functions that can be used within a `@mplang.function` traced graph. The backend implementation for these functions will handle the specific TEE logic.

### 4.1. Frontend API

The `mplang.frontend.tee` module will expose the following functions:

* `gen_nonce() -> MPObject`: Generates a fresh, cryptographic random number on the executing party.
* `quote_gen(report_data: MPObject) -> MPObject`: A core attestation function that can only be executed on a TEE-enabled party. It requests the TEE platform to generate a `quote` that cryptographically binds the machine's identity, the program's identity, and the input `report_data`. If not run in a TEE, it will raise a runtime error.
* `quote_verify(quote: MPObject, ...) -> MPObject`: Verifies a received `quote` against a set of expectations (program hash, data hash, nonces). This is typically executed on non-TEE parties.

Data transmission between parties (e.g., sending a nonce) is handled by MPLang's existing communication primitives, such as `pshfl`, which are used implicitly when data is moved between parties.

### 4.2. Conceptual Execution Flow (Party A's View)

The following shows the logical sequence of operations.

```python
# Phase 1: Distribution & Challenge
# A computation graph is received from the Driver (out of band)
# A calculates graph_hash locally, knows EXPECTED_C_PROGRAM_HASH
graph_hash = hash(computation_graph)
my_nonce = tee.gen_nonce() # Executed on A
# Send my_nonce to C. This is an implicit data movement.
# For example, by using my_nonce in a function that runs on C.

# Phase 2 & 3: Attestation & Verification
# B also sends its nonce to C. C collects all nonces.
# On C, the following is executed:
# report_data = hash(graph_hash, a_nonce, b_nonce)
# c_quote, c_pubkey = tee.quote_gen(report_data)

# The quote is broadcast from C to A and B.
# On A, the following is executed:
is_trusted = tee.quote_verify(c_quote, c_pubkey,
                              EXPECTED_C_PROGRAM_HASH,
                              graph_hash,
                              my_nonce,
                              b_nonce # A needs B's nonce to verify
                             )
assert(is_trusted) # Runtime MUST halt here if false

# Phase 4: Execution
# ... continue with computation ...
```

### 4.3. Example MPLang Implementation

Here is how the flow could be implemented in an MPLang script.

```python
import mplang
import mplang.simp as simp
import mplang.frontend.tee as tee # Proposed TEE frontend

# Parties are identified by an index, e.g., 0, 1, 2
A, B, C = 0, 1, 2

@mplang.function
def secure_computation_with_attestation(a_data, b_data):
    # The traced graph of this function is the "computation_graph"
    graph_hash = "..." # Assume this can be obtained or is a constant

    # 1. Generate nonces on A and B
    a_nonce = simp.runAt(A, tee.gen_nonce)
    b_nonce = simp.runAt(B, tee.gen_nonce)

    # 2. Generate quote on C, implicitly passing nonces
    # The `runAt` specifies where the function executes.
    # MPLang's tracer understands that a_nonce and b_nonce are inputs
    # to this step and must be sent to C.
    c_quote, c_pubkey = simp.runAt(C, tee.quote_gen, graph_hash, a_nonce, b_nonce)

    # 3. Verify on A and B
    # The quote is implicitly broadcast from C to A and B for verification.
    a_is_trusted = simp.runAt(A, tee.quote_verify, c_quote, c_pubkey, graph_hash, a_nonce, b_nonce)
    b_is_trusted = simp.runAt(B, tee.quote_verify, c_quote, c_pubkey, graph_hash, a_nonce, b_nonce)

    # This assertion will be checked by the runtime on A and B respectively
    assert a_is_trusted
    assert b_is_trusted

    # 4. Proceed with actual computation...
    # ...
    return ...
```

**Note:** The exact method for bundling nonces and hashes into `report_data` for `quote_gen` needs to be standardized by the backend implementation. The key is that all verifiers must have all the necessary data to reconstruct the `report_data` for verification.

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

## 6. Implementation Guidance for Runtime

* The `quote_gen` instruction **MUST** be implemented as a call to the underlying platform's TEE SDK (e.g., Intel SGX SDK's `sgx_create_quote`). The runtime is responsible for the IPC with the platform's quoting service daemon (e.g., `aesmd`).
* The `quote_verify` instruction needs access to a trust store containing the TEE vendor's public root certificates. This might be configured globally for the runtime.
* The runtime **MUST** treat the `assert(is_trusted)` step as a critical security boundary. Failure must result in an immediate and clean termination of the computation for that party. No further `trans` or `recv` operations should be possible.
