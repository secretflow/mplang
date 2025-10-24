# MPLang Tutorials

Welcome to MPLang tutorials! This guide introduces two complementary programming models for multi-party computation.

## Two Programming Models

MPLang provides two levels of abstraction for writing multi-party programs:

### 🎯 Device API (Recommended)

The Device API operates on **virtual devices** (PPU, SPU, TEE) rather than physical parties. It provides:

- **Safe cross-device movement**: Automatic encryption/decryption when moving data between devices
- **Coarse-grained control**: Focus on what to compute, not how parties communicate
- **Device attributes**: Framework tracks data placement automatically
- **Type-safe operations**: Compile-time checks for device compatibility

**Use Device API when**:

- Building multi-party applications quickly
- Leveraging existing MPC/TEE backends
- You want automatic security guarantees
- Prototyping and iterative development

**Tutorials**: `device/00_device_basics.py` → `device/05_ir_dump_and_analysis.py`

**Quick start**:

```bash
uv run tutorials/device/00_device_basics.py
```

---

### ⚙️ Simp API (Advanced)

The Simp API (Single Instruction Multiple Parties) exposes **physical party ranks** directly. It provides:

- **Unsafe mode**: "Just do it" - you manage security yourself
- **Fine-grained control**: Direct access to party communication and masks
- **Flexible protocols**: Implement custom multi-party algorithms
- **Performance tuning**: Optimize communication patterns manually

**Use Simp API when**:

- Implementing custom cryptographic protocols
- Optimizing communication at the party level
- Research on multi-party control flow semantics
- Device API cannot express your requirements

**Tutorials**: `simp/00_simp_basics.py` → `simp/05_patterns_and_pitfalls.py`

**Quick start**:

```bash
uv run tutorials/simp/00_simp_basics.py
```

---

## Relationship Between Device and Simp

```text
┌─────────────────────────────────────────────────┐
│  Device API (High-level, Safe)                  │
│  - Virtual devices (PPU, SPU, TEE)              │
│  - Automatic security                           │
│  - Coarse-grained control                       │
└─────────────────┬───────────────────────────────┘
                  │ Built on top of
                  ▼
┌─────────────────────────────────────────────────┐
│  Simp API (Low-level, Unsafe)                   │
│  - Physical party ranks                         │
│  - Manual security management                   │
│  - Fine-grained control                         │
└─────────────────────────────────────────────────┘
```

**Key insights**:

- Device API is **built on top of** Simp API, adding device attributes and safety checks
- You can **mix** Device and Simp code: drop into Simp for custom kernels, return to Device afterward
- Both compile with `@mp.function` - unified compilation pipeline
- Think of Simp as an "unsafe area" where you take full responsibility for correctness

**Recommended workflow**:

1. Start with Device API for application logic
2. Profile and identify bottlenecks
3. Drop into Simp API for custom kernels if needed
4. Return to Device API for remaining logic

---

## Setup

Install dependencies:

```bash
uv sync --group dev
uv pip install -e .
```

Tutorials use mock implementations for SPU/TEE in simulator mode - no external setup required.

---

## Learning Path

**New users**: Start with `device/00_device_basics.py`

**Advanced users**: Explore `simp/` for low-level control

**API Reference**: See `device/index.md` and `simp/index.md` for detailed guides

---

## Resources

- [Design Docs](../design/): Architecture and technical decisions
- [Examples](../examples/): Full applications (XGBoost, neural networks)
- [API Reference](../mplang/): Core modules and functions

Questions? Check track-specific `index.md` files or open an issue.
