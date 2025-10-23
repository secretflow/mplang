# Table ↔ Tensor Structural Conversion (Final Contract)

## Overview

Provide the thinnest possible bridge between a fully‑prepared homogeneous table and a dense 2‑D tensor `[N, F]`.
All semantic preprocessing (projection, casting, encoding, null handling, ordering) must occur upstream (e.g. SQL).
These Frontend Ops are purely structural: they express layout, shape, and column names (analogous to `df.columns`) in MPLang IR without triggering runtime work during tracing.

## Scope (What This Covers)

Included:

- Pack an entire homogeneous table's columns (all columns, in original order) into a tensor.
- Unpack a tensor back into a homogeneous table with explicit column names.
- Enforce invariants (rank, shape, homogeneity, naming arity) at trace time.

Excluded (must be done upstream before calling these ops):

- Column subset / projection
- Per / cross column casting or type normalization
- Null / NA filling or dropping
- Categorification, tokenization, normalization, scaling
- Row filtering, sampling, shuffling, ordering
- Statistics, counting, or shape inference

## API Contract

| FEOp              | fn_type                 | Direction      | Inputs (runtime args)            | Attributes (captured) | Output                                                    |
| ----------------- | ----------------------- | -------------- | -------------------------------- | --------------------- | --------------------------------------------------------- |
| `table_to_tensor` | `basic.table_to_tensor` | Table → Tensor | (table, number_rows:int)         | none                  | `TensorType(d, (N,F))` where d is the shared column dtype |
| `tensor_to_table` | `basic.tensor_to_table` | Tensor → Table | (tensor, column_names:list[str]) | column_names          | `TableType([(name,d) * F])` (names preserved order)       |

Where:

-- `table` is an `MPObject` wrapping a logical table with ordered columns `[c0,…,cF-1]` all sharing the same `dtype d`; column order MUST already be final (mirrors `df.columns`).

- `number_rows (N)` must be a non‑negative Python `int` known at call time.
- `tensor` is an `MPObject` wrapping a rank‑2 tensor of shape `(N,F)` and homogeneous dtype `d`.
- `column_names` length must equal `F` and each name must be a non‑empty unique string.

## Invariants & Validation

`table_to_tensor` rejects:

- Empty schema (no columns)
- Heterogeneous column dtypes
- `number_rows < 0`

`tensor_to_table` rejects:

- Tensor rank ≠ 2
- `column_names` empty / length mismatch with F
- Duplicate or empty column names

Homogeneous dtype is required symmetrically in both directions. No implicit casting is attempted.

## Error Semantics (Summary)

| Op              | Condition            | Exception  |
| --------------- | -------------------- | ---------- |
| table_to_tensor | empty schema         | ValueError |
| table_to_tensor | heterogeneous dtypes | TypeError  |
| table_to_tensor | number_rows < 0      | ValueError |
| tensor_to_table | tensor rank != 2     | TypeError  |
| tensor_to_table | bad / empty names    | ValueError |
| tensor_to_table | name count ≠ F       | ValueError |

## Rationale (Why So Minimal?)

| Choice                   | Why                                                           | Consequence                                        |
| ------------------------ | ------------------------------------------------------------- | -------------------------------------------------- |
| Mandatory `number_rows`  | Prevents hidden COUNT(\*) / ensures static shape for backends | Caller must know N (explicitness over hidden scan) |
| No column projection     | Keeps IR node deterministic & simple                          | Upstream must project                              |
| No casting / encoding    | Separation of concerns; fail fast                             | Upstream adds preprocessing stage                  |
| Homogeneous only         | Simplifies tensor typing; avoids per‑column packing metadata  | Mixed schemas require upstream normalization       |
| Explicit names on unpack | Eliminates guesswork, stable schema reconstruction            | Slight verbosity                                   |

## Backend Behavior

Runtime (pandas DataFrame / numpy ndarray today):

1. `table_to_tensor`: column order preserved exactly as provided (`df.columns`); `np.column_stack([df[c] for c in df.columns])` producing shape `(N,F)`; no casting performed.
2. `tensor_to_table`: constructs DataFrame with provided column names and the tensor's dtype (order stable, one-to-one with second tensor dimension index).

Backends MAY implement equivalent logic with their native arrays; semantics must match ordering and homogeneity constraints.

## Example

```python
# Preprocess upstream (pandas) → homogeneous DataFrame df_h
pfunc_pack, inputs_pack, _ = table_to_tensor(table=df_h_mp, number_rows=len(df_h))

# Later, to reconstruct a logical table
pfunc_un, inputs_un, _ = tensor_to_table(tensor=tensor_mp, column_names=["f1","f2","f3"])
```
