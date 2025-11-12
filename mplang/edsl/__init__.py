# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MPLang EDSL (Embedded Domain-Specific Language) - Experimental Architecture.

This package contains the next-generation EDSL architecture for MPLang,
designed with the following goals:

1. **Modern IR**: Operation List (vs Expr Tree) for better optimization
2. **Unified Type System**: typing.MPType as the single source of truth
3. **Explicit Tracing**: No magic @primitive decorators
4. **Extensibility**: Easy to add new backends (FHE, TEE, etc.)

## Architecture Overview

```
User Code
    ↓
Tracer (explicit)
    ↓
Graph (Operation List + SSA Values)
    ↓
Lowering (to backend IR: MLIR, XLA, etc.)
    ↓
Execution (Simulator, Driver, etc.)
```

## Status

**⚠️ EXPERIMENTAL**: This is a work-in-progress architecture. The stable API
is in `mplang.core`. Use this package only for:

- Contributing to the new architecture
- Testing new features
- Migration experiments

## Migration Path

Old Code (mplang.core):
```python
from mplang import function, peval


@function
def compute(x):
    return x + 1
```

New Code (mplang.edsl):
```python
from mplang.edsl import Tracer, trace

tracer = Tracer(cluster_spec)
graph, outputs = tracer.trace(compute, x)
```

## Design Documents

See `mplang/edsl/design/` for detailed design documents:
- `architecture.md`: Complete EDSL architecture
- `type_system.md`: New type system design
- `migration.md`: Migration strategy

## Current Implementation Status

- [x] Type System (typing.py)
- [ ] Graph IR (graph.py)
- [ ] Builder API (builder.py)
- [ ] Explicit Tracer (tracer.py)
- [ ] Control Flow (primitives/)
- [ ] Integration Tests

## Contributing

This is the cutting edge of MPLang development. If you want to contribute:

1. Read the design docs in `design/`
2. Check the current implementation status above
3. Open an issue/PR to discuss your changes

## See Also

- `mplang.core`: Stable EDSL implementation
- `mplang.ops`: Frontend operations (shared)
- `mplang.kernels`: Backend implementations (shared)
"""

# Type System (ready to use)
from mplang.edsl.typing import (
    HE,
    MP,
    SIMD_HE,
    SS,
    BaseType,
    Custom,
    CustomType,
    MPType,
    ScalarHEType,
    ScalarTrait,
    ScalarType,
    SIMDHEType,
    SSType,
    Table,
    TableType,
    Tensor,
    TensorType,
    f32,
    f64,
    i32,
    i64,
)

__all__ = [
    "HE",
    "MP",
    "SIMD_HE",
    "SS",
    # Type System
    "BaseType",
    "Custom",
    "CustomType",
    "MPType",
    "SIMDHEType",
    "SSType",
    "ScalarHEType",
    "ScalarTrait",
    "ScalarType",
    "Table",
    "TableType",
    "Tensor",
    "TensorType",
    # Scalar types
    "f32",
    "f64",
    "i32",
    "i64",
    # Future: Graph, Tracer, Builder, etc.
]

# Version info
__version__ = "0.1.0-experimental"
