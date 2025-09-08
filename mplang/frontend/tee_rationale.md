# TEE Integration Rationale: Physical vs. Virtual Participants

This document provides a deep analysis of the core design challenge in `mplang`: the distinction between Physical Participants (PP) and Virtual Participants (VP), and how this duality informs the integration of specialized hardware like Trusted Execution Environments (TEE).

## 1. Definitions: Physical vs. Virtual Participants

To build a robust architecture, we must first establish clear definitions for these two concepts within the `mplang` context.

### Physical Participant (PP)

- **Definition**: A Physical Participant is an independent, network-addressable computing instance. In the `SIMP` (Secure Multi-Party) model of `mplang`, it corresponds directly to a process or thread with a unique `rank`.
- **Characteristics**:
  - It is a member of the total set of parties, whose size is given by `psize()`.
  - It is the fundamental unit that constitutes a `pmask` (a mask or set of parties).
  - It is the execution agent for `peval` and the communication endpoint for primitives like `bcast_m` and `scatter_m`.
- **Conclusion**: The `simp` core layer (`mplang.core.primitive`) is designed **exclusively and entirely around the concept of PPs**. Its world consists only of PPs and collections of PPs (`Mask`).

### Virtual Participant (VP)

- **Definition**: A Virtual Participant is a logical abstraction representing a "role" or a "computational domain" created to achieve a specific purpose. It signifies a **computational capability** or a **security boundary**.
- **Characteristics**:
  - A VP can be composed of one or more PPs.
  - Its internal implementation details are opaque to the higher-level caller.
  - It defines a set of operations with specific semantics (e.g., `seal`, `reveal`, `srun`).
- **Examples**:
  - **An SPU Domain**: This is a classic VP. It is composed of multiple PPs (the SPU participants) working in concert. An operation like `srun` is logically applied to the SPU domain (the VP).
  - **Data Owner "Alice"**: This is a VP. In an `mplang` program, we might reason about "Alice's data," while under the hood, Alice might be using the PP with `rank=0` to execute her part of the computation.
  - **A TEE Node**: This is the crucial case, which embodies the core of our design problem.

## 2. `smpc`'s Role: The VP to PP Translator

With these definitions, the role of the `smpc` module (and the proposed `SecureDevice` abstraction) becomes clear:

**`smpc` is a VP-oriented semantic layer that describes "what to do." Its primary responsibility is to translate these high-level semantic goals into a sequence of "how-to" instructions that the `simp` core layer can execute on PPs.**

- **VP Perspective (`spu.seal(data)`)**: "Seal the data `data` into the SPU secure domain."
- **Translated PP Operations**:
    1. `peval` is executed on the PP holding `data` to invoke a `makeshares` function, generating secret shares.
    2. `scatter_m` is used to distribute these shares to all the PPs that constitute the SPU domain.

This translation process is the core value provided by a `SPUDevice` implementation.

## 3. The Core Question: Is a TEE Node a PP or a VP?

The answer is subtle but critical: **a TEE node exhibits characteristics of both, and our architecture must explicitly distinguish and leverage this duality.**

### TEE as a PP (Physical Perspective)

From a networking and deployment standpoint, a TEE node is simply a server running a process. It has an IP address, a port, and can communicate with other `mplang` participants. Therefore, **at the `SIMP` infrastructure layer, treating the TEE node as a PP (e.g., by assigning it a `rank`) is not only reasonable but necessary.** This is how other PPs will send encrypted data to it using `simp` primitives.

### TEE as a VP (Logical Perspective)

From a security and computation standpoint, a TEE provides a **single, atomic, trusted black-box execution service**. Unlike SPU, whose computation is inherently **distributed** (multiple PPs collaborating), a TEE's computation is **centralized** (performed independently inside the TEE's secure enclave).

When we `seal` data to a TEE, our intent is to move it into the "TEE" **security domain (VP)**, not just to the process at `rank=N` (the PP). Our trust is in the technological guarantees of the TEE, not in the specific process.

### The Contradiction and Its Resolution

The central conflict arises from the `SIMP` model's SPMD (Single Program, Multiple Data) assumption, where all PPs are expected to execute the same program logic. A TEE node fundamentally breaks this assumption:

- **Regular PPs (e.g., `rank=0, 1`)**: Execute the main `mplang` logic written by the user.
- **TEE PP (e.g., `rank=2`)**: Does not execute the main logic. Its sole purpose is to act as a service: wait for encrypted data, execute a function within its enclave, and return an encrypted result.

Simply treating the TEE node as a standard PP would break the SPMD model. You cannot instruct the TEE PP to participate in an `spu.srun` alongside regular PPs; the operation is meaningless for it.

## 4. Conclusion and Architectural Design Recommendation

The architecture must resolve this by decoupling the TEE's dual identity.

**The `simp` layer should only be aware of the TEE's PP identity, while the `smpc` (or `device`) layer must abstract it into a VP.**

### Design Specifics

1. **Infrastructure Layer (`runtime`)**:
    - During startup, the TEE node joins the `SIMP` network as a standard PP and is assigned a `rank`.
    - However, it runs a specialized `driver` or `server` process, not the general-purpose `mplang` script executor. This server exposes TEE-specific RPCs like `run_in_enclave`.

2. **`smpc` Abstraction Layer**:
    - Define a `TEEDevice(SecureDevice)`.
    - This `TEEDevice` is initialized with the `rank` of the TEE node.

    ```python
    class TEEDevice(SecureDevice):
        def __init__(self, tee_rank: Rank):
            self.tee_rank = tee_rank
            self.tee_mask = Mask.from_ranks(tee_rank)

        def seval(self, fe_type: str, pyfn: Callable, *args, **kwargs):
            # 1. Compile/serialize the function and its arguments into a
            #    format the TEE can understand (e.g., a specific MLIR dialect).
            payload = self._compile_for_tee(pyfn, *args, **kwargs)

            # 2. Encrypt the payload and send it to the TEE PP.
            #    'obj.owner' refers to the PP holding the data.
            encrypted_payload = prim.peval(
                encrypt_for_tee, [payload], obj.owner
            )
            prim.bcast_m(self.tee_mask, obj.owner, encrypted_payload)

            # 3. Trigger the execution on the TEE PP. This is a special
            #    peval that only executes a "call enclave" primitive on the tee_mask.
            encrypted_result = prim.peval(
                self._run_in_enclave_primitive, [], self.tee_mask
            )

            # 4. Retrieve and decrypt the result...
            return result
    ```

### Essence of the Design

- **Separation of Concerns**: The implementation of `TEEDevice.seval` is concerned with PP-level operations (`bcast_m`, `peval` on `tee_mask`). However, the interface it exposes to the user (`seval`) is a clean, VP-level black-box operation.
- **Asymmetric SPMD**: The design acknowledges that the TEE PP's behavior is **asymmetric**. By carefully orchestrating `peval` calls that execute on specific `pmask`s, the `TEEDevice` implementation cleverly works around the "all PPs do the same thing" limitation to achieve asymmetric logic.
- **Clean Abstraction**: The user simply instantiates `spu = SPUDevice(...)` or `tee = TEEDevice(rank=2)` and calls `spu.seval()` or `tee.seval()`. The user interacts with a consistent VP abstraction, while the underlying PP-level complexity is perfectly encapsulated.

### Summary

A TEE node is physically a PP but must be logically abstracted as a VP. The key to this abstraction is a dedicated `Device` implementation (`TEEDevice`) that translates VP-level semantic operations into a series of asymmetric `simp` primitive calls targeting specific subsets of PPs. This approach clarifies the layered relationship between `simp` and `smpc` and provides a robust architectural pattern for integrating participants with specialized, non-uniform behaviors.
