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

### 4.1. Cluster Configuration and Context Initialization

MPLang requires explicit cluster configuration and runtime context setup.
This makes cluster definition a **first-class citizen** and eliminates
implicit state issues.

```python
import mplang as mp

# Step 1: Define cluster (from dict or YAML file)
cluster_spec = mp.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:8100"},
        {"name": "node_1", "endpoint": "127.0.0.1:8101"},
    ],
    "devices": {
        "P0": {"kind": "PPU", "members": ["node_0"]},
        "P1": {"kind": "PPU", "members": ["node_1"]},
        "SP0": {"kind": "SPU", "members": ["node_0", "node_1"],
                "config": {"protocol": "SEMI2K", "field": "FM128"}},
    },
})

# Step 2: Create runtime context (Simulator or Driver)
sim = mp.make_simulator(2, cluster_spec=cluster_spec)  # Local testing
# or
driver = mp.make_driver(cluster_spec.endpoints, cluster_spec=cluster_spec)  # Distributed

# Step 3: Use context manager or set as global
with sim:
    result = my_function()
# or
mp.set_root_context(sim)  # JAX-like global context pattern
```

This pattern:

- Loads and validates the cluster configuration via `ClusterSpec`
- Creates an execution context (Simulator for local, Driver for distributed)
- Provides explicit control over context lifetime
- Enables both context manager and global context patterns

### 4.2. D-API: `mplang.device` (Device-Level API)

**Target Audience**: Data scientists, algorithm engineers, application developers.

**Philosophy**: High-level, secure, and oriented towards logical devices. It
completely hides `rank` and other physical details. This is the **recommended
API for 90% of users**.

**Core APIs**:

- `device(dev_id)` - decorator/context for device-specific execution
- `put(dev_id, data)` - place data on a device
- `fetch(obj)` - retrieve data from device to host
- `get_dev_attr(obj)` / `set_dev_attr(obj, dev_id)` - manage device attributes

**Example**:

```python
import jax.numpy as jnp
import mplang as mp

# Devices are referenced by string ID from cluster config
# Device IDs: "P0", "P1", "SP0", "TEE0", etc.

# Place data on devices
x = mp.put("P0", jnp.array([1, 2, 3]))
y = mp.put("P1", jnp.array([4, 5, 6]))

# Use device decorator for computation
# For PPU (plaintext): use .jax property for JAX functions
@mp.device("P0").jax
def local_computation(data):
    return data * 2  # Uses JAX operators

# For SPU (secure MPC): JAX is native, no .jax needed
@mp.device("SP0")
def secure_computation(a, b):
    return a + b  # Executed securely via MPC

# Execute computations
result_local = local_computation(x)
result_secure = secure_computation(x, y)

# Fetch results
final_result = mp.fetch(result_secure)

# Auto device inference (infers from argument devices)
@mp.device()
def auto_infer(a, b):
    return a + b

result_auto = auto_infer(x, y)  # Infers device from x, y
```

### 4.3. P-API: `mplang.dialects.simp` (Party-Level API)

**Target Audience**: Framework developers, security protocol researchers,
advanced users needing fine-grained control.

**Philosophy**: Low-level, flexible, and oriented towards physical parties/ranks.
It provides maximum control but **offers no built-in security guarantees**.
Users are responsible for the correctness and security of their code.

**Core SIMP Primitives** (`mplang.dialects.simp`):

- `pcall_static(parties, fn, *args)` - execute function on explicit parties
- `pcall_dynamic(fn, *args)` - execute on all parties (runtime-determined)
- `shuffle_static(src, routing)` - data redistribution with static routing
- `shuffle_dynamic(src, index)` - data redistribution with runtime index
- `converge(*vars)` - merge disjoint partitions
- `uniform_cond(pred, then_fn, else_fn, *args)` - uniform conditional
- `while_loop(cond_fn, body_fn, *init)` - while loop
- `constant(parties, value)` - create constant on specific parties

**Collective Operations** (`mplang.libs.collective`):

- `transfer(data, to=rank)` - point-to-point transfer
- `replicate(data, to=parties)` - broadcast to multiple parties
- `distribute(values, frm=rank)` - scatter from one to many
- `collect(data, to=rank)` - gather from many to one
- `allreplicate(data)` - allgather-like operation

**Example**:

```python
from mplang.dialects import simp
from mplang.libs import collective

# Execute on specific parties (rank-based)
x = simp.pcall_static((0,), lambda: 42)

# Data redistribution with static routing
y = simp.shuffle_static(x, routing={1: 0})  # Send to party 1

# High-level collective operations
z = collective.replicate(x, to=(0, 1, 2))  # Broadcast

# Control flow
result = simp.uniform_cond(
    pred,
    then_fn=lambda: "yes",
    else_fn=lambda: "no"
)
```

### 4.4. Domain Libraries: `mplang.libs`

**Philosophy**: Provide high-level protocols and operations for common
multi-party computation tasks, built on top of SIMP primitives.

**Available Libraries**:

**`mplang.libs.collective`** - MPI-style collective communication:
- `transfer`, `replicate`, `distribute`, `collect`, `allreplicate`, `permute`

**`mplang.libs.mpc`** - Multi-party computation protocols:
- `mpc.psi` - Private Set Intersection (PSI) protocols
  - `psi_intersect` (RR22), `psi_unbalanced`, `eval_oprf`
- `mpc.ot` - Oblivious Transfer protocols
  - `transfer` (Naor-Pinkas), `transfer_extension` (IKNP), `silent_vole_random_u`
- `mpc.vole` - Vector OLE protocols
  - `silver_vole`, `gilboa_vole`
- `mpc.analytics` - Privacy-preserving analytics
  - `oblivious_groupby_sum_bfv`, `apply_permutation`, `secure_switch`

**`mplang.libs.ml`** - Machine learning:
- `ml.sgb` - Secure Gradient Boosting

**Example**:

```python
from mplang.libs.collective import replicate, collect
from mplang.libs.mpc.psi import psi_intersect
from mplang.libs.mpc.ot import transfer as ot_transfer
from mplang.dialects import simp

# Collective operations
data = simp.constant((0,), [1, 2, 3])
replicated = replicate(data, to=(0, 1, 2))

# Private Set Intersection
intersection = psi_intersect(set_a, set_b, sender=0, receiver=1)

# Oblivious Transfer
messages = [msg0, msg1]
chosen = ot_transfer(messages, choice_bit, sender=0, receiver=1)
```

### 4.5. API Usage Guidelines

**Recommended Patterns**:

1. **D-API First**: Start with `mplang.device` API for most applications
   - Use `mp.put()` to place data, `@mp.device("P0")` for execution
   - PPU functions need `.jax` property: `@mp.device("P0").jax`
   - SPU functions use JAX natively: `@mp.device("SP0")`

2. **P-API for Protocols**: Use SIMP primitives when building custom protocols
   - `simp.pcall_static` for explicit party execution
   - `simp.shuffle_static/dynamic` for data redistribution
   - `collective.*` for high-level communication patterns

3. **Domain Libraries**: Leverage `mplang.libs.*` for common tasks
   - `libs.collective` for communication patterns
   - `libs.mpc.psi` for set intersection
   - `libs.mpc.ot` for oblivious transfer
   - `libs.ml.sgb` for secure gradient boosting

4. **Context Management**: Always use explicit context
   ```python
   sim = mp.make_simulator(world_size, cluster_spec=cluster_spec)
   with sim:
       result = my_function()
   ```

5. **Avoid Mixing**: Don't mix device API and raw SIMP in the same function
   unless you understand the party mask implications

## 5. Conclusion

This architectural redesign provides a robust, pragmatic foundation for MPLang .
The new structure achieves:

- **Clarity**: Clear separation between logical devices (D-API) and physical parties (P-API)
- **Usability**: 90% of users can use the simple, safe D-API without thinking about ranks
- **Flexibility**: Expert users have full control through the powerful P-API
- **Maintainability**: Clean module structure with explicit responsibilities
- **Evolutionary**: Minimal disruption to existing code through strategic refactoring

**Key Benefits**:

1. **Explicit Configuration**: `ClusterSpec` makes cluster definition a first-class object
2. **Layered APIs**: Clear separation between device-level (D-API) and party-level (P-API)
3. **Rich Protocol Libraries**: Extensive `mplang.libs.*` for PSI, OT, VOLE, analytics
4. **Flexible Contexts**: Support both context managers and global context patterns
5. **Future-Proof**: Easy to add new device types and protocols without architectural changes

The implementation can be achieved through gradual refactoring rather than
revolutionary rewriting, making this a practical path forward for the MPLang ecosystem.
