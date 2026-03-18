# Workload Schema

A **Workload** makes a Definition concrete and executable by binding specific values to all variable axes and specifying where the input data comes from. It is the exact configuration under which a Solution is benchmarked.

The evaluator runs your solution against every Workload in the problem's set — typically 16–48 workloads per problem.

---

## Table of Contents
1. [Top-Level Schema](#top-level-schema)
2. [inputs — Input Descriptors](#inputs--input-descriptors)
3. [tolerance — Per-Workload Correctness Bounds](#tolerance--per-workload-correctness-bounds)
4. [Examples](#complete-example--gemm-with-random-inputs)
5. [How the Evaluator Uses Workloads](#how-the-evaluator-uses-workloads)

---

## Top-Level Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `uuid` | string | Yes | Unique identifier for this workload |
| `axes` | object | Yes | Map of `var` axis names → concrete integer values |
| `inputs` | object | Yes | Map of input names → input descriptor (how to source the data) |
| `tolerance` | object | No | Per-workload correctness bounds (defaults below) |

**Storage:** In the dataset, all workloads for one definition are stored as a JSONL file (one JSON object per line).

---

## `inputs` — Input Descriptors

Each input descriptor specifies where the tensor data comes from. The `type` field selects the sourcing strategy.

### `type: "random"`

Generate a random tensor using the shape and dtype from the Definition. No additional fields needed.

```json
"A": {"type": "random"},
"B": {"type": "random"}
```

### `type: "scalar"`

A fixed Python scalar value passed directly to the function (for inputs with `"shape": null` in the Definition).

```json
"eps": {"type": "scalar", "value": 1e-6},
"temperature": {"type": "scalar", "value": 0.7}
```

| Field | Required | Description |
|-------|----------|-------------|
| `value` | Yes | `int`, `float`, or `bool` scalar |

### `type: "safetensors"`

Load a specific tensor from a `.safetensors` file. Used for inputs with real-world data (e.g. KV-cache indices, weights from actual model runs).

```json
"kv_indptr": {
  "type": "safetensors",
  "path": "blob/workloads/gqa_paged/b1_s512.safetensors",
  "tensor_key": "kv_indptr"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | Relative path to the `.safetensors` file |
| `tensor_key` | Yes | Key inside the safetensors container |

### `type: "custom"`

Inputs that the Definition's `custom_inputs_entrypoint` function generates. The evaluator calls that function to produce these tensors instead of generating them randomly. This allows problem-specific bounded data.

```json
"router_indices": {"type": "custom"},
"expert_weights":  {"type": "custom"}
```

No additional fields required — the shape and generation logic come from `custom_inputs_entrypoint` in the Definition's `reference` code.

**When you see `"type": "custom"` in a workload:** the evaluator will call `definition.custom_inputs_entrypoint(axes_and_scalars, device)` to produce these inputs. Your `run()` function receives them as normal tensors — you don't need to handle this differently.

---

## `tolerance` — Per-Workload Correctness Bounds

Each workload can specify its own tolerance for correctness checking via the `tolerance` field. If omitted, defaults apply.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_atol` | float | `1e-2` | Maximum absolute error threshold |
| `max_rtol` | float | `1e-2` | Maximum relative error threshold |
| `required_matched_ratio` | float | `0.99` | Fraction of elements that must pass the tolerance bound |
| `max_error_cap` | float \| null | `null` | Hard ceiling on max absolute error (fails regardless of matched ratio) |
| `allow_negative_inf` | bool | `false` | When true, matching `-inf` values in both output and reference are treated as correct and excluded from error computation |

**Correctness formula** (torch.allclose style): An element passes if `|output - reference| <= max_atol + max_rtol * |reference|`. The workload passes if at least `required_matched_ratio` of all elements pass AND (if `max_error_cap` is set) the largest absolute error is below the cap.

**Negative infinity handling:** When `allow_negative_inf` is `true`, positions where both output and reference are `-inf` are treated as correct and excluded from the error computation. Positions where only one tensor has `-inf` still fail. `+inf` and `NaN` are unaffected by this flag.

```json
"tolerance": {
  "max_atol": 1e-3,
  "max_rtol": 1e-3,
  "required_matched_ratio": 0.95,
  "max_error_cap": 0.1
}
```

**Note:** Tolerance is per-workload, not global. Different workloads for the same problem can have different tolerance specs. This allows tighter bounds for easy shapes and looser bounds for numerically challenging configurations.

---

## Complete Example — GEMM with Random Inputs

```json
{
  "uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "axes": {"M": 1024},
  "inputs": {
    "A": {"type": "random"},
    "B": {"type": "random"}
  },
  "tolerance": {"max_atol": 1e-3, "max_rtol": 1e-3}
}
```

## Complete Example — GQA with Mixed Inputs

```json
{
  "uuid": "6120f144-b973-4bd9-b884-77ecb132914e",
  "axes": {
    "B": 4,
    "Q": 512,
    "KV": 2048,
    "H_qo": 32,
    "H_kv": 8
  },
  "inputs": {
    "q":         {"type": "random"},
    "k":         {"type": "safetensors", "path": "blob/gqa/b4.safetensors", "tensor_key": "k"},
    "v":         {"type": "safetensors", "path": "blob/gqa/b4.safetensors", "tensor_key": "v"},
    "kv_indptr": {"type": "safetensors", "path": "blob/gqa/b4.safetensors", "tensor_key": "kv_indptr"}
  }
}
```

## Complete Example — MoE with Custom Inputs

```json
{
  "uuid": "deadbeef-0000-1111-2222-333344445555",
  "axes": {"T": 256},
  "inputs": {
    "hidden_states":  {"type": "random"},
    "router_indices": {"type": "custom"},
    "expert_weights": {"type": "custom"}
  }
}
```

Here `hidden_states` is random, but `router_indices` and `expert_weights` have structural constraints (valid expert indices, valid probability weights) so they are generated by `custom_inputs_entrypoint`.

---

## How the Evaluator Uses Workloads

For each workload, the evaluator:

1. Binds `var` axes to their concrete values from `workload.axes`
2. Generates/loads each input tensor according to its descriptor:
   - `random` → `torch.randn(shape, dtype=dtype, device=device)` (or appropriate random for the dtype)
   - `scalar` → the literal value
   - `safetensors` → loads tensor from file
   - `custom` → calls `definition.custom_inputs_entrypoint(axes_and_scalars, device)`
3. Pre-allocates output tensors (shapes from Definition.outputs with resolved axes)
4. Calls `solution.run(*inputs, *outputs)` — inputs in Definition order, then outputs
5. Compares outputs against `reference(*inputs)` for correctness
6. Times the solution for performance
