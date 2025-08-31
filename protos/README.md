## Project Architecture

MPLang follows a modular architecture that separates API definitions from implementation:

### Current Structure (Development Mode)

```
mplang/
├── mplang/                      # Core library implementation
├── protos/                      # API definitions (.proto files)
├── mplang/protos/               # Generated Python bindings
└── buf.yaml                     # Buf v2 workspace configuration (replaces buf.work.yaml)
```

> **Note on `mplang-proto`**: This directory contains protobuf definitions that can function as a standalone repository supporting multi-language integration. However, due to early development stage, it's currently embedded within the mplang repository for rapid iteration. Generated Python code is placed in `mplang/mplang_proto`. The mplang package contains two top-level directories: `mplang` and `mplang_proto`.
