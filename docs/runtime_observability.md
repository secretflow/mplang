# MPLang Runtime Observability Guide

This guide explains how to use debugging and observability features in MPLang v2.

## Overview

MPLang provides three key observability features:

1. **Persistence Layout**: Structured directory layout for cache, store, and trace outputs
2. **Execution Tracing**: Chrome Trace JSON output for performance analysis
3. **Graph Printing**: Save execution graph IR to file for debugging

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MPLANG_DATA_ROOT` | Root directory for all persistence | `.mpl` |
| `MPLANG_PRINT_GRAPH` | Save execution graph to file | `false` |

---

## Persistence Layout

### Directory Structure

```
${MPLANG_DATA_ROOT}/
└── <cluster_id>/
    ├── __host__/
    │   ├── trace/      # Host-side traces
    │   └── graphs/     # Saved graph IR files
    └── node<N>/
        ├── cache/jax/  # JAX compilation cache
        ├── store/      # ObjectStore persistence
        └── trace/      # Worker-side traces
```

### Cluster IDs

| Backend | Cluster ID |
|---------|-----------|
| `SimpSimulator(world_size=3)` | `__sim_3` |
| `SimpHttpDriver` | (uses `__host__` directly) |

---

## Execution Tracing

### Enabling Tracing

```python
import mplang.v2 as mp

# For Simulator (local)
sim = mp.Simulator.simple(3, enable_tracing=True)

# For CLI
# python -m mplang.v2.cli sim -f job.py --profile
```

### Viewing Traces

1. Open Chrome → `chrome://tracing`
2. Load JSON from `.mpl/__sim_N/__host__/trace/*.json`

### Key Classes

| Class | Purpose |
|-------|---------|
| `ExecutionTracer` | Records events in Chrome Trace format |
| `_NullTracer` | No-op stub when disabled |

---

## Graph Printing

### Usage

```bash
MPLANG_PRINT_GRAPH=1 python -m mplang.v2.cli sim -f examples/v2/psi_bench.py
```

### Output

```
[Graph] Saved to .mpl/__sim_2/__host__/graphs/graph_<job_id>.txt
```

### Example Graph IR

```mlir
%0 = simp.pcall_static() {fn_name='constant', parties=[0]} : MP[Tensor[...]] {
    () {
      %0 = tensor.constant() {...} : Tensor[u64, (100)]
      return %0
    }
  }
%1 = simp.shuffle(%0) {dst_rank=1} : MP[...]
```

---

## Worker Sandbox

Each worker operates in an isolated "sandbox" directory:

```python
worker = WorkerInterpreter(
    rank=0,
    world_size=3,
    communicator=comm,
    tracer=tracer,
    root_dir="/data/mplang/cluster_abc/node0"  # Mandatory
)
```

---

## Best Practices

1. **Use `root_dir`**: Don't hardcode paths. Use `interpreter.root_dir / "subdir"`.
2. **Cluster isolation**: Different configurations should use different `cluster_id` values.
3. **Production**: Disable tracing (`enable_tracing=False`) for production workloads.
4. **Cache sharing**: JAX cache can be shared across jobs within the same cluster.
