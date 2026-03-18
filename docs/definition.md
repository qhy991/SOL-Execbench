# Definition Schema

A **Definition** provides the formal specification for a computational workload — the operator's contract. It defines the axes, tensor shapes/dtypes, optional constraints, and a correct PyTorch reference implementation.

**Identity rule:** Two kernels share a Definition if and only if they have the same axes with the same roles (`const`/`var`) and the same `const` values.

---

## Table of Contents
1. [Top-Level Schema](#top-level-schema)
2. [axes — Dimension Definitions](#axes--dimension-definitions)
3. [inputs / outputs — Tensor Definitions](#inputs--outputs--tensor-definitions)
4. [reference — PyTorch Implementation](#reference--pytorch-implementation)
5. [custom_inputs_entrypoint](#custom_inputs_entrypoint)
6. [Examples](#examples)

---

## Top-Level Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique human-readable name. Convention: `{op_type}_{props}_{constants}` (e.g. `gqa_paged_decode_h32_kv8_d128_ps1`) |
| `op_type` | string | No* | Compute category. e.g. `gemm`, `rmsnorm`, `gqa_ragged`, `gqa_paged` |
| `description` | string | No | Brief human-readable description |
| `axes` | object | Yes | Map of symbolic axis names → `AxisConst`, `AxisVar`, or `AxisExpr` |
| `inputs` | object | Yes | Named input tensors |
| `outputs` | object | Yes | Named output tensors |
| `reference` | string | Yes | PyTorch reference implementation (Python source as string) |
| `constraints` | array[string] | No | Assertions relating axes (e.g. `"H_qo == H_kv * H_r"`) |
| `custom_inputs_entrypoint` | string | No | Function name in `reference` that generates custom inputs — see below |

*`op_type` is optional in the server dataset (older entries may not have it).

---

## `axes` — Dimension Definitions

Each key is a symbolic dimension name (e.g. `"M"`, `"batch_size"`, `"hidden_size"`). Value is one of:

### `type: "const"` — Fixed at problem definition time

```json
"hidden_size": {
  "type": "const",
  "value": 4096,
  "description": "Model hidden dimension"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | `"const"` |
| `value` | Yes | Fixed integer value |
| `description` | No | Brief description |

### `type: "var"` — Determined by each Workload at runtime

```json
"batch_size": {
  "type": "var",
  "description": "Number of sequences in the batch"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | `"var"` |
| `description` | No | Brief description |

### `type: "expr"` — Computed from other axes

```json
"total_tokens": {
  "type": "expr",
  "expression": "batch_size * seq_len",
  "description": "Total number of tokens"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | `"expr"` |
| `expression` | Yes | Mathematical expression referencing other axis names. Supported operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`, parentheses, and unary `+`/`-`. |
| `description` | No | Brief description |

---

## `inputs` / `outputs` — Tensor Definitions

Each key is a tensor name (matching what your `run()` function receives). Value:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `shape` | array or `null` | Yes | List of axis name strings. `[]` = 0-D tensor. `null` = Python scalar |
| `dtype` | string | Yes | Data type (see dtype values below) |
| `description` | string | No | Brief description |

### Supported `dtype` values

`float64`, `float32`, `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1`, `float4_e2m1fn_x2`, `int64`, `int32`, `int16`, `int8`, `bool`

### Scalars vs 0-D Tensors

- `"shape": null` → Python scalar (`int`, `float`, `bool`) passed directly
- `"shape": []` → 0-dimensional `torch.Tensor`
- `"shape": ["dim1", "dim2"]` → standard tensor

---

## `reference` — PyTorch Implementation

A Python source string containing the mathematical specification of the operator. Rules:
- Must define a global function named `run` as entry point
- Use explicit step-by-step PyTorch (avoid `torch.nn.functional` shortcuts — prefer clarity)
- Returns the output tensors

```python
# Example reference for RMSNorm
"import torch\n\ndef run(input, weight, eps):\n    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)\n    rstd = torch.rsqrt(variance + eps)\n    hidden_states = input * rstd\n    output = (hidden_states * weight).to(weight.dtype)\n    return output"
```

---

## `custom_inputs_entrypoint`

**Field:** `custom_inputs_entrypoint: Optional[str]`

When set, names a function defined in the `reference` code that generates problem-specific bounded inputs. The evaluator calls this instead of generating random tensors for inputs marked `"type": "custom"` in a Workload.

**Why it exists:** Some inputs have structural constraints that pure random generation can't satisfy:
- Router indices (must be valid indices bounded by `num_experts`)
- TopK selection outputs (bounded output of a sampling op)
- Softmax outputs (must sum to 1 across a dimension)
- Ragged sequence length arrays (must be non-decreasing, sum to total tokens)

**Function signature:**

```python
def fn(axes_and_scalars: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]:
    """
    Args:
        axes_and_scalars: All const axis values + scalar inputs from the workload.
                          Keys are axis names (str), values are Python ints.
        device: The target CUDA device.
    Returns:
        Dict mapping input names (matching Definition.inputs keys) to generated tensors.
        Only needs to return the inputs that are marked 'custom' in the workload.
    """
```

**Example:**

```json
{
  "name": "moe_dispatch_h4096_e64_k4",
  "custom_inputs_entrypoint": "generate_inputs",
  "axes": {
    "T": {"type": "var"},
    "H": {"type": "const", "value": 4096},
    "E": {"type": "const", "value": 64},
    "K": {"type": "const", "value": 4}
  },
  "inputs": {
    "hidden_states": {"shape": ["T", "H"], "dtype": "bfloat16"},
    "router_indices": {"shape": ["T", "K"], "dtype": "int32"}
  },
  "reference": "import torch\n\ndef generate_inputs(axes_and_scalars, device):\n    T = axes_and_scalars['T']\n    E = axes_and_scalars['E']\n    K = axes_and_scalars['K']\n    # router_indices must be valid expert indices\n    router_indices = torch.randint(0, E, (T, K), device=device, dtype=torch.int32)\n    return {'router_indices': router_indices}\n\ndef run(hidden_states, router_indices, output):\n    ..."
}
```

---

## Examples

### Standard GEMM

```json
{
  "name": "gemm_n4096_k4096",
  "op_type": "gemm",
  "description": "General matrix multiply C = A @ B.T",
  "axes": {
    "M": {"type": "var"},
    "N": {"type": "const", "value": 4096},
    "K": {"type": "const", "value": 4096}
  },
  "inputs": {
    "A": {"shape": ["M", "K"], "dtype": "float16"},
    "B": {"shape": ["N", "K"], "dtype": "float16"}
  },
  "outputs": {
    "C": {"shape": ["M", "N"], "dtype": "float16"}
  },
  "reference": "import torch\n\ndef run(A, B):\n    return torch.matmul(A, B.T)"
}
```

### RMSNorm with Scalar Input

```json
{
  "name": "rmsnorm_h4096",
  "op_type": "rmsnorm",
  "axes": {
    "batch_size": {"type": "var"},
    "hidden_size": {"type": "const", "value": 4096}
  },
  "inputs": {
    "input":  {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
    "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
    "eps":    {"shape": null, "dtype": "float32"}
  },
  "outputs": {
    "output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}
  },
  "reference": "import torch\n\ndef run(input, weight, eps):\n    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)\n    rstd = torch.rsqrt(variance + eps)\n    return (input * rstd * weight).to(weight.dtype)"
}
```

### GQA Attention with Constraints

```json
{
  "name": "gqa_ragged_hr4_dqk128_dvo128",
  "op_type": "gqa_ragged",
  "axes": {
    "B":   {"type": "var"},
    "Q":   {"type": "var"},
    "KV":  {"type": "var"},
    "H_qo":{"type": "var"},
    "H_kv":{"type": "var"},
    "H_r": {"type": "const", "value": 4},
    "D_qk":{"type": "const", "value": 128},
    "D_vo":{"type": "const", "value": 128}
  },
  "constraints": ["H_qo == H_kv * H_r"],
  "inputs": {
    "q": {"shape": ["B", "Q", "H_qo", "D_qk"], "dtype": "float16"},
    "k": {"shape": ["B", "KV", "H_kv", "D_qk"], "dtype": "float16"},
    "v": {"shape": ["B", "KV", "H_kv", "D_vo"], "dtype": "float16"}
  },
  "outputs": {
    "out": {"shape": ["B", "Q", "H_qo", "D_vo"], "dtype": "float16"},
    "lse": {"shape": ["B", "Q", "H_qo"], "dtype": "float32"}
  },
  "reference": "..."
}
```
