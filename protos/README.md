## Project Architecture

MPLang follows a modular architecture that separates API definitions from implementation:

### Current Structure (Development Mode)

```
mplang/
├── buf.yaml                      # Buf v2 workspace configuration
├── buf.gen.yaml                  # Code generation configuration
├── protos/                       # API definitions (.proto files)
│   └── mplang/protos/v1alpha1/
│       └── mpir.proto
├── mplang/                       # Core Python library
│   └── protos/                   # Generated Python bindings (after buf generate)
│       └── v1alpha1/
└── ...
```

> Note: The protobuf definitions can live in a standalone repo for multi-language integration.
> During early development, they are embedded here for faster iteration.
> Generated Python code is placed under `mplang/protos/v1alpha1/` after generation.
