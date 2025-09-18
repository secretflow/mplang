# MPLang Frontend Architecture (Redesign Proposal)

This document proposes a modern, extensible, and reflective frontend (FE) architecture for MPLang that
unifies existing frontends (JAX, Ibis, builtin), supports module-level state safely, and enables future
expansion (e.g., new dialects, capabilities, and device routing) without tight coupling.

We explicitly do not consider backward compatibility here; the design favors clarity and extensibility first.

## Goals

- Single, consistent FE model across JAX/Ibis/Builtin/others.
- Clean separation between tracing/IR creation vs execution/runtime.
- Deterministic IR capture regardless of runtime caches.
- First-class state model: module-level (process-wide) and context-local (cur_ctx-root) with guardrails.
- Introspection/reflection APIs (list ops, capabilities, config schemas).
- Device routing driven by FE metadata, not isinstance.
- Pluggable/additive: new FE ops register via a registry with minimal boilerplate.
- Testable contracts: determinism, purity boundaries, capability matching.

Non-goals:

- Re-implementing backends or runtime here.
- Enforcing 100% source compatibility with current modules (we’ll provide a migration path).

## Core Abstractions

### FE Operation (feop)

- A feop is a callable that returns an IR-building result (usually a PFunction and its inputs/trees) or triggers higher-level peval composition.
- Decorated with `@feop(...)` to attach metadata, state hooks, and register in the FE registry.
- Works for both function compilers (e.g., JAX compile) and special ops (e.g., SPU make_shares/reconstruct).

Proposed decorator shape (conceptual):

```python
@feop(
  name="jax.compile",
  family="jax",               # family/namespace for routing
  dialects=["pphlo"],         # produced IR dialect(s)
  version="1",
  config_schema=JaxConfigSchema, # optional pydantic-like schema
  state_scope="ctx",          # "ctx" | "process" | "none"
  side_effects=False,          # indicates IR purity
)
def jax_compile(fn, *a, **kw):
    ...
```

Attached attributes on the callable:

- `__feop__ = True`
- `__feop_meta__` dict with fields above and extra capabilities (see below)
- `__get_state__(ctx) -> dict`: returns a namespaced dict for state storage according to `state_scope`

Helper:

- `is_feop(x) -> bool`: returns True if callable is a feop.

### State Model

Two-tier state with strict rules:

- Process-level state (module-level caches):
  - Only store pure Python artifacts safe to share across contexts/process forks (e.g., MLIR bytes, stable signatures, simple LRU indexes).
  - Never store MPObject, Interp/Trace vars, or context-bound handles.
  - Access via `__get_state__(None)` when `state_scope="process"` or via a module helper `fe_process_state(key)`.

- Context-local state (`ctx.root()`):
  - Store runtime-bound values: MPObjects, sessions, device-local ephemeral keys, etc.
  - Access via `__get_state__(ctx)`; the decorator will derive a per-op namespace key and attach a dict under
    `ctx.root()._fe_state`.
  - Must not affect traced IR shape; it may influence runtime-only behavior (e.g., reusing a TEE session or using
    a cached peval result) but IR determinism must hold.

Guidelines:

- If a state would change compiled IR, surface it as explicit config/args.
- Context state may be seeded lazily per-use.
- Process state should be bounded (LRU) to avoid unbounded growth.

### Metadata and Capabilities (Frontend-wide)

Metadata informs dispatch, validation, reflection, and tooling. Keep it generic across all FEs.

Core fields:

- `name`: canonical dotted name (e.g., "jax.compile", "ibis.compile", "builtin.cond").
- `family`: source/frontend family (e.g., "jax", "ibis", "builtin", "crypto").
- `op_kind`: semantic category, one of { "compiler", "transform", "special_op" }.
- `dialects`: produced IR dialect(s) (e.g., ["pphlo"], ["sql"], ["conv"], ["stablehlo"]).
- `version`: semantic string for evolution (e.g., "1").
- `execution_model`: how the op is expected to run:
  - { "single_rank", "multi_party", "in_controller" }.
  - guides device routing beyond families.
- `inputs`: IO constraints (kinds and optional schemas), e.g.,
  - `[{ kind: "Tensor|Table|Scalar|Bytes", visibility?: ["PUBLIC","PRIVATE"], schema?: Type }]`.
- `outputs`: IO constraints symmetric to inputs.
- `config_schema`: optional schema (e.g., pydantic-like) for config objects.
- `side_effects`: whether IR build has observable side effects.
- `targets`: hint of compatible device kinds, e.g., ["PPU","TEE","SPU"].
- `capabilities`: dictionary for advanced constraints:
  - `requires_device`: constraints like { kind: ["SPU","TEE"], members: "=1|>=1" }.
  - `supports_visibility`: ["PUBLIC","PRIVATE"], etc.
  - `requires_world_size`: numeric constraints.
  - `input_types` / `output_types`: richer type constraints.
  - `stability`: { level: "experimental|stable", notes?: str }.
  - `caching`: { scope: "none|process|ctx", key_strategy?: "signature|custom" }.

Device routing should rely on `execution_model`, `targets`, `dialects`, and `capabilities`, not on `isinstance`.
For example:

- Multi-party devices (e.g., SPU) prefer ops with `execution_model == "multi_party"` and compatible dialects (e.g., `pphlo`).
- Single-rank devices (PPU/TEE) prefer `single_rank` ops or controller-side ops that wrap via `runAt`.
- Builtin control-flow ops (`op_kind == "transform"`) are compatible with any device but may be evaluated in-controller.

### Registry and Reflection

A singleton registry holds all feops by name and by family.

APIs:

- `fe.register(op)`
- `fe.list_ops(family: str | None = None) -> list[FeOpInfo]`
- `fe.get(name: str) -> callable`
- `fe.find(family: str, predicate: callable) -> list[callable]`
- `fe.capabilities(op) -> dict`
- `fe.config_schema(op) -> type | None`

Optional plugin mechanism:

- Use Python entry points (e.g., `mplang.frontends`) to auto-discover external feops.
- Registry is populated at import time or lazily.

### Determinism & Contracts

A feop must satisfy:

- Deterministic IR: Given (fn/args/config), the produced IR (e.g., PFunction IR, plus metadata) is stable and
  reproducible across runs.
- Purity boundary: If `side_effects=False`, the op must not depend on hidden state for IR; caches only accelerate
  identical results.
- Type contract: op declares its input expectations and output shapes/dtypes abstractions (or validates at runtime).
- Error taxonomy: raise `FeConfigError`, `FeCapabilityError`, `FeDeterminismError` when contracts are violated.

### Compilation Pipeline (example: JAX)

Standard stages for compiler-type feops:

1. Normalize fn and arguments to a canonical form (strip closures when possible, ShapeDtypeStruct for leaves).
2. Build a stable signature key (hash of normalized function text/bytecode + pytree spec + shapes/dtypes + config
  options).
3. Check process-level compile cache (if enabled) to reuse MLIR bytes or PFunction skeleton.
4. Compile by calling FE module/compiler (e.g., libspu, XLA, Ibis SQLgen) to IR bytes.
5. Wrap IR in PFunction + metadata (names, visibilities, dialect, version) and return.
6. Optionally pre-validate capabilities against device intents (early errors).

This yields consistent behavior and cross-device portability.

## Device Routing Integration

Replace `isinstance(op, FEOp)` checks with metadata-driven rules:

- Compute: `meta = getattr(op, "__feop_meta__", {})`
- For SPU devices: allow `meta["family"] == "jax"` (or another whitelist), and optionally `"pphlo" in meta["dialects"]`.
- For TEE devices: allow `family in {"jax", "ibis"}` etc.
- For PPU: generally allow all families but enforce member-count constraints if any.

Routing then executes the op in the appropriate placement (e.g., `simp.runAt` for single-rank devices, `smpc.srun` for
SPU/JAX combos) guided by capabilities, not by Python class hierarchies.

## Return Contract (unified)

Most feops should have a single, simple return model: they materialize an IR program description that the runtime can
evaluate independently of Python callables.

We standardize on the following shapes:

- FeProgram (recommended for the majority of ops):
  - pfunc: PFunction (IR)
  - in_tree: PyTreeDef for inputs
  - out_tree: PyTreeDef for outputs (optional if derivable)
  - meta: dict (e.g., names, visibilities, dialect, version)

- FeSignature (optional building block):
  - A canonical description of expected inputs (pytree of abstract leaves such as ShapeDtypeStruct) and optional
    constraints (visibility, names). This allows compiler-like ops to be deterministic without seeing real data.

Compiler-like outliers (e.g., jax.compile) have two integration options:

1) Explicit-signature compile (preferred):
   - Inputs: (fn, signature: FeSignature, config?)
   - Return: FeProgram
   - This avoids changing the feop call pattern for others and guarantees determinism at build time.

2) Two-phase builder (optional convenience):
   - feop returns a FeBuilder (callable) that, when given args, derives a signature and produces a FeProgram.
   - Provide a normalization utility `fe_materialize(op, *args, **kwargs) -> FeProgram` so routers can uniformly obtain
     a program. This path is syntactic sugar and should not be the primary contract.

Guidelines:

- Favor the explicit-signature compile for compiler-style feops. Offer a thin convenience helper outside the feop API
  (e.g., `jax_compile_from_data(fn, *args)`) for ergonomic cases; it derives a signature then calls the feop.
- Non-compiler feops should directly return FeProgram based on the provided typed inputs.
- Determinism: FeProgram must be a pure function of (fn/FeSignature/config/meta); runtime data must not influence IR.

## State Examples

- TEE session caching (runtime-only):
  - Stored in `ctx.root()._fe_state[("tee", "sessions")]` as pure bytes/keys or MPObjects.
  - Access via a device helper (as today) or a feop using `__get_state__(ctx)` if the feop owns the state.

- JAX compile cache (process-level):
  - Keyed by signature; values are MLIR bytes and simple metadata.
  - Safe across contexts, but never store MPObjects.

- SPU per-config ops:
  - Implement as callable instances with config; feop decorator attaches metadata and a `__get_state__`.
  - Context state might hold IR-to-backend adapter hints if needed (still not MPObjects).

## Reflection & Tools

- `mplang.frontend.introspect` module:
  - `list_frontends() -> list[str]` families
  - `list_ops(family: str | None) -> list[FeOpInfo]`
  - `describe(op) -> dict` including schema, capabilities, routes, examples

- `mplang.analysis` integration:
  - Visualize feops used in a program (graph-level metadata stamps per feop call sites).

## Error Taxonomy

- `FeError`: base class.
- `FeConfigError`: invalid or missing config; schema violations.
- `FeCapabilityError`: op cannot run with target device capabilities.
- `FeDeterminismError`: non-deterministic IR detected for same signature.
- `FeStateError`: illegal state usage (e.g., MPObject stored in process state).

## Testing & Conformance

- Determinism test: compiling the same fn+shapes/config twice yields bit-identical IR (or checksum match).
- Capability test: invalid device routing raises `FeCapabilityError` with clear message.
- State purity test: process-level caches forbid MPObjects by type-check.
- Reflection test: every exported feop is present in registry with metadata.

## Sketch: Minimal Decorator and Registry

```python
# frontend/base.py

_REGISTRY = {}
_PROCESS_STATE = {}

def feop(**meta):
    def deco(fn):
        fn.__feop__ = True
        fn.__feop_meta__ = meta
        key = (meta.get("family"), meta.get("name", fn.__name__))
        _REGISTRY[key] = fn

        def __get_state__(ctx=None):
            scope = meta.get("state_scope", "none")
            if scope == "none":
                return {}
            if scope == "process":
                return _PROCESS_STATE.setdefault(key, {})
            # ctx scope
            if ctx is None:
                raise FeStateError("ctx state requires a context")
            root = ctx.root()
            if not hasattr(root, "_fe_state"):
                root._fe_state = {}
            return root._fe_state.setdefault(key, {})

        fn.__get_state__ = __get_state__
        return fn
    return deco

def is_feop(x):
    return bool(getattr(x, "__feop__", False))
```

This is illustrative; production code should include typing, defensive checks, and docs.

## Migration Path (High-Level)

1. Implement `feop` decorator, `is_feop`, registry, and minimal reflection APIs.
2. Port JAX/Ibis compilers to feops with proper metadata and process-level caches.
3. Port SPU special ops (make_shares/reconstruct) to feops with `family="spu"` or keep them under `family="builtin"` with explicit `capabilities`.
4. Update device routing to use metadata (family/dialects/capabilities).
5. Provide shims from legacy modules (`jax_cc.jax_compile`, etc.) to new feops for gradual adoption.
6. Add conformance tests for determinism and reflection.

## Appendix: Why This Design

- Avoids brittle `isinstance` coupling; favors metadata and contracts.
- Supports both stateless and stateful feops with clear scoping rules.
- Encourages pure, deterministic IR construction with optional caching.
- Scales to external plugins (e.g., new SQL engines, custom dialects) via registry.
- Aligns with MPLang’s context discipline (no cross-context MPObject leakage).
