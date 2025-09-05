# Design Doc: Verifiable Computation Primitives

**Status:** Draft
**Owner:** [Your Name/Team]
**Last Updated:** September 5, 2025

## 1. Motivation & Goal

Our Multi-Party Computation IR (MPC-IR) needs to support scenarios where some parties (`A`, `B`) don't trust each other but are willing to trust a third party `C` running in a TEE (Trusted Execution Environment).

The goal of this design is to introduce a minimal set of IR primitives and a corresponding runtime protocol that allows `A` and `B` to **cryptographically verify** that `C` is a genuine TEE instance, running the **correct program**, and executing the **exact same IR script** they have, before they send any sensitive data to it.

This document outlines the "how" and "why" to guide the implementation of the IR and Runtime.

## 2. High-Level Flow

The entire process is orchestrated by a `Driver` which distributes the IR. The flow is as follows:

1. **Distribution:** The `Driver` sends an identical `IR_script` to all participants (`A`, `B`, and `C`).
2. **Challenge & Attestation:** `A` and `B` challenge `C` to prove its identity and its commitment to the received `IR_script`. `C` responds with a cryptographic proof (`quote`).
3. **Verification:** `A` and `B` verify this proof.
4. **Execution:** If verification succeeds, all parties proceed with the secure computation defined in the `IR_script`.

This is a **blocking** process. If verification fails for any party, the entire computation must halt.

## 3. Trust & Verification Logic

### 3.1. Pre-existing Trust Assumptions

Our design relies on the following foundational trust relationships:

* **Trust in Business Logic:** All participants (`A`, `B`, `C`) trust the `IR_script` they receive from the `Driver`. The mechanism for this trust (e.g., `Driver` digitally signing the IR) is **out of scope** for this protocol but is a prerequisite. All parties start with a shared, trusted `IR_script`.
* **Trust in TEE Hardware:** `A` and `B` trust the TEE hardware manufacturer (e.g., Intel, AMD) and have access to their public root certificates. This is a fundamental platform assumption.

### 3.2. Who Proves What to Whom?

This is the core logic of the protocol:

* **The Prover:** The TEE-based party `C`.
* **The Verifiers:** The non-TEE parties `A` and `B`.
* **The Proof's Content (`C` proves to `A` and `B` that...)**:
    1. **"I am a real TEE."** (Authenticity): This is proven by a cryptographic signature on the `quote` that chains back to the TEE hardware manufacturer's trusted root certificate.
    2. **"I am running the program you expect."** (Code Integrity): The `quote` contains a measurement of `C`'s program binary (e.g., `MRENCLAVE`). `A` and `B` will check this against a known-good value.
    3. **"I am about to execute the exact same IR script you have, right now."** (Logic & Freshness): This is the most critical part. The `quote` must also contain a commitment to the specific `IR_script` for this session, combined with a freshness guarantee.

The combination of these three proofs establishes a secure context for the computation.

## 4. IR Primitives & Runtime Requirements

To implement this flow, we need to introduce a minimal set of new IR instructions. Note that low-level crypto operations (`Encrypt`, `Hash`, etc.) are assumed to be provided by a standard library; these are the special, protocol-level instructions.

### 4.1. Minimal IR Instruction Set

| Instruction        | Description                                                                                                                                              | Execution Environment    |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------- |
| `gen_nonce()`      | Generates a fresh, cryptographic random number.                                                                                                          | Any                      |
| `quote_gen(data)`  | The core attestation primitive. It requests the TEE platform to generate a `quote` that cryptographically binds the machine's identity, the program's identity, and the input `data`. | **Must be in a TEE**     |
| `quote_verify(...)`| Verifies a received `quote` against a set of expectations (program hash, data hash, nonces).                                                              | Any (typically non-TEE)  |
| `trans(dest, data)`| A pre-existing primitive for sending data to another party.                                                                                             | Any                      |
| `recv(src)`        | A pre-existing primitive for receiving data.                                                                                                             | Any                      |

### 4.2. Example IR Execution Flow (Party A's View)

```ir
# Phase 1: Distribution & Challenge
# IR_script is received from Driver (out of band)
# A calculates ir_hash locally, knows EXPECTED_C_PROGRAM_HASH
ir_hash = hash(IR_script)
my_nonce = gen_nonce()
trans('C', my_nonce)

# Phase 2 & 3: Attestation & Verification
# B also sends its nonce to C. C generates and broadcasts one quote.
c_quote, c_pubkey = recv('C')
is_trusted = quote_verify(c_quote, c_pubkey,
                          EXPECTED_C_PROGRAM_HASH,
                          ir_hash,
                          my_nonce,
                          # Must also get B's nonce to verify the full quote_data
                          b_nonce # Assume B broadcasts its nonce or C includes it in the response
                         )
assert(is_trusted) # Runtime MUST halt here if false

# Phase 4: Execution
my_data = ...
encrypted_data = encrypt(c_pubkey, my_data)
trans('C', encrypted_data)

# ... continue with computation ...
```

**Note:** The exact method for exchanging `b_nonce` needs to be defined (e.g., `C` can bundle it in its response to `A`). The key is that all verifiers must have all the nonces to reconstruct the `quote_data` for verification.

## 5. Security Analysis: Attack Vectors & Defenses

This design is intended to counter the following specific attacks:

### 5.1. Attack: Malicious `C` Impersonation

* **Vector:** An attacker runs a non-TEE program that pretends to be the trusted party `C`.
* **Defense:** `quote_verify` will fail at the very first step. The attacker cannot produce a `quote` with a valid signature that chains back to a TEE hardware vendor.

### 5.2. Attack: Wrong Program Execution

* **Vector:** A genuine TEE `C` is running the wrong version of the program (e.g., an old, vulnerable version).
* **Defense:** `quote_verify` will fail. The `program_hash` inside the `quote` will not match the `EXPECTED_C_PROGRAM_HASH` that `A` and `B` expect.

### 5.3. Attack: IR Mismatch (Logic Deception)

* **Vector:** The `Driver` (or a Man-in-the-Middle) sends a different `IR_script` to `C` than to `A` and `B`.
* **Defense:** `quote_verify` will fail. `C`'s `quote` will contain a commitment to `hash(evil_IR)`, while `A` and `B` will be expecting a commitment to `hash(good_IR)`. The verification of `quote_data` will fail.

### 5.4. Attack: Replay Attack

* **Vector:** An attacker records a valid `quote` from a previous, legitimate session and replays it to `A` and `B` to trick them into sending data.
* **Defense:** This is what the nonces are for. `A` and `B` generate **fresh, unpredictable nonces** for *every single session*. The `quote` is cryptographically bound to these specific nonces. A replayed `quote` will contain old nonces and will be rejected by `quote_verify`. This ensures the proof is "live".

## 6. Implementation Guidance for Runtime

* The `quote_gen` instruction **MUST** be implemented as a call to the underlying platform's TEE SDK (e.g., Intel SGX SDK's `sgx_create_quote`). The runtime is responsible for the IPC with the platform's quoting service daemon (e.g., `aesmd`).
* The `quote_verify` instruction needs access to a trust store containing the TEE vendor's public root certificates. This might be configured globally for the runtime.
* The runtime **MUST** treat the `assert(is_trusted)` step as a critical security boundary. Failure must result in an immediate and clean termination of the computation for that party. No further `trans` or `recv` operations should be possible.
