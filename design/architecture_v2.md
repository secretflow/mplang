# MPLang Architecture v2: Design Document

**Version:** 1.0
**Date:** 2025-09-08
**Status:** Proposed

## 1. Introduction & Motivation

This document outlines a significant architectural redesign for MPLang. The current
architecture, while functional, presents several challenges:

- **Implicit State:** Modules like `smpc` rely on an implicit, context-based
  state (`spu_mask`), making behavior less predictable and harder to test.

- **Leaky Abstractions:** Low-level concepts like `rank` are exposed in high-level
  APIs (e.g., `smpc.sealFrom`), creating a confusing programming model.

- **Ambiguous Participant Roles:** The distinction between a physical compute node
  and a logical computation domain is not clearly defined, making it difficult to
  model complex scenarios like TEE integration.

This redesign aims to establish a clear, robust, and scalable architecture by
introducing a strict separation between physical and logical concepts.

## 2. Core Philosophy: Physical vs. Logical Worlds

The new architecture is built upon a fundamental duality:

- **Physical World (The "How")**: Composed of **Physical Nodes** (also referred to as Physical Parties or **PPs**). A Node is an
  actual, addressable compute process with a unique `rank`. It is the ultimate
  executor of computation and communication. The P-API operates exclusively
  in this world.

- **Logical World (The "What")**: Composed of **Logical Devices** (also referred to as Virtual Parties or **VPs**). A Device is a
  user-facing entity representing a specific computational capability or security
  domain (e.g., an SPU computation, a TEE enclave). A Device is realized by one
  or more Nodes.

The core responsibility of the framework is to translate user intentions expressed
in the Logical World into concrete actions executed in the Physical World.

## 3. Cluster Configuration (`cluster.yaml`)

To reflect the PP/VP duality, the cluster configuration is redesigned into two
distinct top-level sections.

### 3.1. `nodes`: The Physical Layer

This section is a list that serves as the **single source of truth** for the
physical world. Each element is a physical node with the following attributes:

- `name`: A unique, human-readable identifier.
- `rank`: The unique integer ID. **It is implicitly defined by the node's order
  (index) in the `nodes` list**, which simplifies configuration and prevents errors.
- `endpoint`: The network address of the node process.
- `runtime_info` (optional): Metadata about the node's capabilities, such as
  `version`, supported `backends` (`spu`, `tee_enclave_runner`,
  `filesystem_io`), and `platform`.

### 3.2. `devices`: The Logical Layer

This section is a dictionary defining the complete universe of logical devices
addressable by the D-API. Each key is a logical device name.

- `kind`: The type of device. We define several base kinds:
  - `local`: A device representing a single physical node for local computation.
  - `spu`: A composite device for multi-party computation.
  - `tee`: A device for trusted execution environments (e.g., Intel SGX, ARM TrustZone).
- `members`: A list of `name`s from the `nodes` section, linking the device
  to its constituent physical nodes.
- `config`: Device-specific configuration (e.g., SPU protocol, enclave binary path).

### 3.3. Example Configuration

The `cluster.yaml` file in this directory serves as a concrete reference,
demonstrating the power of the new structure by clearly modeling a scenario
where Carol's capabilities are split into two separate physical nodes for security.

## 4. API Design: Two-Layer Architecture

The new MPLang architecture provides two distinct, clearly demarcated API layers
that cater to different user needs and use cases.

### 4.1. Initialization: `mplang.init()` as First-Class Citizen

Before using either API layer, users must initialize MPLang with a cluster
configuration. This makes cluster definition a **first-class citizen** in the
MPLang namespace and solves all implicit state issues.

```python
import mplang

# Initialize with cluster definition (file path or dict)
mplang.init("path/to/cluster.yaml")
# or
mplang.init(cluster_dict)
```

This single initialization step:

- Loads and validates the cluster configuration
- Creates a global context accessible by both API layers
- Enables device discovery and validation
- Makes all subsequent API calls stateless and predictable

### 4.2. D-API: `mplang.device` (Device-Level API)

**Target Audience**: Data scientists, algorithm engineers, application developers.

**Philosophy**: High-level, secure, and oriented towards logical devices. It
completely hides `rank` and other physical details. This is the **recommended
API for 90% of users**.

**Core APIs**:

- `device` decorator to make a function execute on a specific Logical Device
- `to` moves data from one device to another, handling security concerns automatically

**Example**:

```python
# Get logical devices from cluster config
alice = mplang.device("alice")        # kind: local
bob = mplang.device("bob")            # kind: local
spu = mplang.device("spu_3pc")        # kind: spu
tee = mplang.device("trusted_party")  # kind: tee

# Place data on devices
x_data = [1, 2, 3]
y_data = [4, 5, 6]
x_on_alice = mplang.to(alice, x_data)
y_on_bob = mplang.to(bob, y_data)

# Use devices as decorators for computation
@alice
def local_computation(data):
    return data * 2

@spu
def secure_computation(a, b):
    return a + b  # Executed securely across parties

# Execute computations
result_local = local_computation(x_on_alice)
result_secure = secure_computation(x_on_alice, y_on_bob)

# Fetch results
final_result = mplang.fetch(result_secure)
```

### 4.3. P-API: `mplang.simp` (Party-Level API)

**Target Audience**: Framework developers, security protocol researchers,
advanced users needing fine-grained control.

**Philosophy**: Low-level, flexible, and oriented towards physical parties/ranks.
It provides maximum control but **offers no built-in security guarantees**.
Users are responsible for the correctness and security of their code.

**Core APIs**:

- `run`, `run_at` executes functions on specific Physical Nodes by rank
- `bcast`, `scatter`, `gather`, `allgather` provide MPI-style communication primitives
- `cond`, `while_loop`, `pconv` enable low-level control flow programming
- `seal`, `reveal`, `srun` offer direct SMPC operations without abstraction
- `pshfl`, `pshfl_s` provide secure shuffling capabilities

**Example**:

```python
from mplang import simp

# Execute on specific physical party (rank)
obj_ref = simp.run_at(0, lambda: "hello from alice_node")

# Broadcast data across all parties
all_objs = simp.bcast(obj_ref, root=0)

# Use low-level SMPC operations
sealed_obj = simp.seal(obj_ref)
result = simp.reveal(sealed_obj)
```

### 4.4. Universal Libraries: `mplang.lib`

**Philosophy**: Provide computation primitives that work seamlessly across both
API layers with **context-aware behavior**.

**Core APIs**:

- Context-aware Libraries: `random`

**Context-Aware Behavior**:

```python
from mplang.lib import random

# Same random API, different execution contexts:

# In D-API: automatically uses secure random when in SPU context
@spu
def secure_random():
    return random.rand((2, 3))  # → SPU secure random

@alice
def local_random():
    return random.rand((2, 3))  # → numpy.random

# In P-API: direct control over execution location
simp.run_at(0, lambda: random.rand((2, 3)))  # → numpy.random on rank 0
```

### 4.5. API Usage Guidelines

**Recommended Patterns**:

1. **D-API First**: Start with D-API for most applications
2. **No Mixing**: Avoid mixing D-API and P-API in the same function unless you
   understand the implications
3. **Library Consistency**: Use `mplang.lib.*` for computations that need to
   work across different execution contexts

## 5. Conclusion

This architectural redesign provides a robust, pragmatic foundation for MPLang v2.
The new structure achieves:

- **Clarity**: Clear separation between logical devices (D-API) and physical parties (P-API)
- **Usability**: 90% of users can use the simple, safe D-API without thinking about ranks
- **Flexibility**: Expert users have full control through the powerful P-API
- **Maintainability**: Clean module structure with explicit responsibilities
- **Evolutionary**: Minimal disruption to existing code through strategic refactoring

**Key Benefits**:

1. **No More Implicit State**: `mplang.init()` makes cluster configuration explicit
2. **Context-Aware Libraries**: Universal libraries that adapt to execution context
3. **Clear API Boundaries**: Users know whether they're working at device or party level
4. **Future-Proof**: Easy to add new device types without architectural changes

The implementation can be achieved through gradual refactoring rather than
revolutionary rewriting, making this a practical path forward for the MPLang ecosystem.
