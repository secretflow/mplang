# Control Flow Guide (Simple Classification → `uniform_cond` → Notes)

Scope: How to pick the correct control-flow pattern in MPLang single-controller SPMD, when to use `uniform_cond`, and related performance / safety notes.

---

## 1. Classification Matrix (Static vs Dynamic × Uniform vs Divergent × Granularity)

Dimensions:

* Static / Dynamic – compile-time constant vs runtime computed
* Uniform / Divergent – same across all parties vs may differ per party
* Granularity – Structural (short-circuit) vs Elementwise (compute both then blend).

| Static/Dynamic | Uniform? | Granularity (Structural / Elementwise) | Recommended | Notes |
|----------------|----------|----------------------------------------|-------------|-------|
| Static | Uniform | Structural or N/A | Python if | Pruned before tracing |
| Static | Divergent | N/A | (unsupported) | Anti-pattern: should be unified pre-trace |
| Dynamic | Uniform | Structural (local) | peval + lax.cond | Both branches local ops |
| Dynamic | Uniform | Structural (multi-party) | uniform_cond | Only construct that skips side-effects |
| Dynamic | Uniform | Elementwise | peval + jax.where | Tensor blending |
| Dynamic | Divergent | Elementwise | compute both branches + where blend | Divergence only data-level |
| Dynamic | Divergent | Structural (desired) | refactor: aggregate to uniform or use data path | Need all-reduce to form uniform scalar |

Decision Flow

```text
Start
 ├─ Static? → Yes:
 │    ├─ Per-party difference? → Yes: Anti-pattern (unify before trace) → End
 │    └─ No → Python if → End
 └─ Static? → No (Dynamic):
     ├─ Uniform? → No (Divergent):
     │    ├─ Need structural short-circuit? → Yes: Unsupported (aggregate to uniform) → continue
     │    └─ No → compute both branches + where blend (elementwise) → End
     └─ Uniform? → Yes:
         ├─ Elementwise blending? → Yes: where → End
         └─ No (Structural):
             ├─ Both branches pure local & light? → Yes: lax.cond / expr → End
             └─ No → uniform_cond
                 ├─ predicate may diverge? → Yes: all-reduce aggregate → uniform_cond
                 └─ No → End
```

---

## 2. `uniform_cond` (Structured Multi-Party Conditional)

### 2.1 When to Use

Dynamic + Uniform + Structural + at least one branch performs multi-party side-effects (communication / exchange / protocol / cross-device transfer / cooperative randomness).

### 2.2 Semantics

`uniform_cond(pred, then_fn, else_fn, *args, verify_uniform=True)`

| Aspect | Rule |
|--------|------|
| Predicate | uniform boolean scalar (`MPObject`) |
| Execution | Only chosen branch executes; zero multi-party side-effects in unchosen branch |
| Output | PyTree structure + each leaf MPType identical (shape/dtype/pmask/device) across branches |
| Capture | Branch closures re-captured in current context (recapture) to ensure validity |
| Verification | Runtime O(P^2) manual all-gather uniformity check; fail → `ValueError` |
| Errors | Output mismatch → `TypeError`; non-uniform predicate → `ValueError` |

### 2.3 Value

* Skip expensive unchosen side-effects / allocations
* Backend may do dead-branch elimination / lazy allocation
* Explicit sync point aiding analysis & visualization

### 2.4 Comparison (lax.cond / where)

| Dimension | uniform_cond | lax.cond | where |
|----------|--------------|----------|-------|
| Single-side multi-party side-effects | Yes | No (both local compute) | No (both compute data) |
| Elementwise blending | No | No | Yes |
| Strict output MPType parity | Yes | No (generic shape/dtype rules) | N/A |

### 2.5 Examples

```python
def heavy_secure(x):
    y = smpc.seal(x)
    return smpc.reveal(y) + constant(1)

def fast_local(x):
    return x - constant(1)

pred = some_uniform_flag  # uniform bool scalar across parties
out = uniform_cond(pred, heavy_secure, fast_local, x)
```

Lightweight local difference (anti-pattern for uniform_cond):

```python
res = uniform_cond(pred, lambda v: v + 1, lambda v: v - 1, x)  # anti-pattern
# Better:
res = peval(jax.lax.cond)(pred, lambda _: x + 1, lambda _: x - 1, operand=None)
```

Elementwise blending:

```python
blend = where(mask, a, b)  # mask may be uniform or divergent
```

Divergent predicate (must compute both):

```python
hi = expensive_local(x)
lo = cheap_local(x)
res = where(divergent_pred, hi, lo)
```

### 2.6 IR / Typing Constraint Example

```python
def bad(a, b):
    def t(x, y): return x   # pmask P0
    def e(x, y): return y   # pmask P1
    return uniform_cond(pred, t, e, a, b)  # -> TypeError: pmask mismatch
```

### 2.7 Verification Implementation (Brief)

Current: O(P^2) point-to-point fan-out/fan-in boolean gather + compare. Future: boolean all-reduce (AND) lowering complexity. `verify_uniform` flag lets advanced users skip when statically guaranteed.

---

## 3. Notes (Performance / Safety / FAQ / Future)

### 3.1 Performance

| Item | Impact |
|------|--------|
| Skip unchosen side-effects | Less protocol / network / memory cost |
| Trace size | Both branches still captured (visualization / typing) |
| Verification overhead | Single short boolean aggregation; can be disabled/optimized |

### 3.2 Safety / Semantics

| Point | Explanation |
|-------|------------|
| Uniform requirement | Prevent mis-pruning on divergent predicates |
| No divergent structural branching | Divergence must go through data path (dual compute + where) |
| Predicate granularity | Only scalar boolean; elementwise needs where |

### 3.3 FAQ

**Q: Can we trace only the taken branch?** Not yet; dual-branch tracing stabilizes typing & analysis; lazy trace is future work.
**Q: Skip verification if I know it's uniform?** Set `verify_uniform=False` (if API exposed); recommended to keep during development.
**Q: Multi-way switch?** Use nested uniform_cond for now; add switch if real demand emerges.
**Q: Want structural pruning with divergent predicate?** Aggregate via all-reduce to uniform scalar first; otherwise elementwise path.

### 3.4 Future Work

| Direction | Description | Status |
|-----------|-------------|--------|
| Boolean all-reduce verification | Reduce O(P^2) to O(log P) | Planned |
| Static uniform inference | Provenance analysis to skip verification | Planned |
| Lazy branch tracing | Trace only chosen branch (cache & backfill) | Research |
| Multi-way switch | Richer structural control form | Evaluating |
