# Solution Schema

A **Solution** is your concrete kernel implementation for a given Definition. It encapsulates the source code, build metadata, and submission identity.

---

## Table of Contents
- [Solution Schema](#solution-schema)
  - [Table of Contents](#table-of-contents)
  - [Top-Level Schema](#top-level-schema)
  - [`spec` — Build Specification](#spec--build-specification)
  - [Destination Passing Style (DPS)](#destination-passing-style-dps)
  - [`compile_options` — Compiler Flags](#compile_options--compiler-flags)
  - [`dependencies` — Supported Libraries](#dependencies--supported-libraries)
  - [`sources` — Source Code Files](#sources--source-code-files)
  - [Supported Languages](#supported-languages)
    - [Python / Triton / CuTe DSL / cuTile Function Signature](#python--triton--cute-dsl--cutile-function-signature)
    - [CUDA C++ Function Signature](#cuda-c-function-signature)
    - [CuTe DSL Best Practices](#cute-dsl-best-practices)
  - [Complete Examples](#complete-examples)
    - [Python Solution](#python-solution)
    - [Triton Solution](#triton-solution)
    - [CUDA C++ Solution](#cuda-c-solution)
    - [CuTe DSL Solution](#cute-dsl-solution)

---

## Top-Level Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique solution name (e.g. `rmsnorm_triton_b200_v2`) |
| `definition` | string | Yes | The `name` of the Definition this solves |
| `author` | string | Yes | Agent or human author identifier |
| `description` | string | No | Brief description of the technique |
| `spec` | object | Yes | Build specification — see below |
| `sources` | array | Yes | Source code files — see below |

---

## `spec` — Build Specification

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `languages` | array[string] | Yes | Programming languages — see Supported Languages. C++ and Python languages cannot be mixed. |
| `target_hardware` | array[string] | Yes | Target hardware list: `["B200"]`, `["local"]`, or both |
| `entry_point` | string | Yes | `"filename::function_name"` — the function the evaluator calls |
| `destination_passing_style` | bool | No | Whether to use DPS — outputs passed as last args (default: `true`) |
| `binding` | string \| null | No | Binding type for C++/CUDA solutions. Defaults to `"torch"` for C++/CUDA languages, ignored for Python. |
| `dependencies` | array[string] | No | Required packages (limited set — see Dependencies) |
| `compile_options` | object | No | Extra compiler flags for CUDA/C++ — see below |

---

## Destination Passing Style (DPS)

DPS is the default calling convention. Set `destination_passing_style: true` (or omit — it defaults to `true`).

The evaluator pre-allocates all output tensors and passes them as the **last positional arguments** after all inputs. Your function writes results in-place and **does not return anything**.

```python
# ✅ DPS = true (required): outputs pre-allocated, write in-place
def run(input, weight, eps, output):
    #                        ^^^^^^ pre-allocated output tensor
    output[:] = normalize(input, weight, eps)
    # no return

# ❌ DPS = false: value-returning style
def run(input, weight, eps):
    return normalize(input, weight, eps)
```

**Argument order:** inputs in the order they appear in `Definition.inputs`, then outputs in the order they appear in `Definition.outputs`.

```python
# For Definition with:
#   inputs:  {A, B, eps}
#   outputs: {C, stats}
def run(A, B, eps, C, stats):  # inputs first, then outputs
    ...
```

---

## `compile_options` — Compiler Flags

For C++ language solutions (e.g. `"cuda_cpp"`, `"cutlass"`), extra compiler flags merged onto the evaluator's defaults:

```json
"compile_options": {
  "cflags":      ["-O3"],
  "cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr"],
  "ld_flags":    []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cflags` | array[string] | Extra GCC/Clang C compiler flags |
| `cuda_cflags` | array[string] | Extra NVCC flags |
| `ld_flags` | array[string] | Extra linker flags |

These are **additive** — do not replicate the evaluator's default flags.

---

## `dependencies` — Supported Libraries

The supported dependency set is limited to:

| Library | Example specifier |
|---------|------------------|
| CUTLASS | `"CUTLASS_3_7"` → injects CUTLASS 3.7 header paths |
| CuTe DSL | `"cutlass"` |
| Triton | `"triton >= 2.3"` |
| cuBLAS | `"cublas"` |
| cuDNN | `"cudnn"` |
| PyTorch | `"torch"` |

Python package specifiers are validated against the current environment — if a listed version is not installed, the build fails fast.

---

## `sources` — Source Code Files

Array of `{path, content}` objects. The evaluator reconstructs this directory structure at evaluation time.

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | Relative file path including extension (e.g. `kernel.py`, `src/kernel.cu`) |
| `content` | Yes | Complete source code text |

**Rules:**
- `path` must be relative (no leading `/`, no `..`)
- The entry point file (`spec.entry_point` prefix) must be in `sources`
- No duplicate paths

---

## Supported Languages
These are the languages used for computation. More than one may be specified, but C++ and Python cannot be mixed.

| `spec.languages` | Entry point format | Notes |
|------------------|-------------------|-------|
| `["pytorch"]` | `"file.py::function_name"` | PyTorch, any pure Python library |
| `["triton"]` | `"file.py::function_name"` | Triton kernels + PyTorch |
| `["cute_dsl"]` | `"file.py::function_name"` | NVIDIA CuTe DSL (Python); treated as Python by evaluator |
| `["cutile"]` | `"file.py::function_name"` | NVIDIA cuTile Python DSL |
| `["cudnn_frontend"]` | `"file.py::function_name"` | NVIDIA cuDNN frontend Python DSL |
| `["cuda_cpp"]` | `"file.cu::function_name"` | CUDA C++; compiled via PyTorch `torch` binding |
| `["cutlass"]` | `"file.cpp::function_name"` | Pure C++ with CUTLASS; compiled via PyTorch `torch` binding |
| `["cudnn"]` | `"file.cpp::function_name"` | C++ with cuDNN; compiled via PyTorch `torch` binding |
| `["cublas"]` | `"file.cpp::function_name"` | C++ with cuBLAS; compiled via PyTorch `torch` binding |

### Python / Triton / CuTe DSL / cuTile Function Signature

```python
# Parameter names must exactly match Definition.inputs keys, then Definition.outputs keys (DPS)
def run(input_name_1, input_name_2, ..., output_name_1, output_name_2, ...):
    # write results into output tensors
    output_name_1[:] = ...
```

`**kwargs` in the signature is allowed and ignored during validation.

### CUDA C++ Function Signature

Uses PyTorch `torch` binding. Entry point is a C function symbol exposed to Python:

```cpp
// Inputs as torch::Tensor by value, outputs as torch::Tensor by reference (written in-place)
void run(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // kernel implementation
    // write result into C in-place
}
```

### CuTe DSL Best Practices

```python
import cutlass.cute as cute
from cuda.core.experimental._dlpack import from_dlpack

# Cache compiled kernel at module level — avoids recompilation per workload
_compiled_fn = None

def run(A, B, C):
    global _compiled_fn
    if _compiled_fn is None:
        # mark_layout_dynamic() makes shape-agnostic JIT — reuses across workloads
        a_cute = from_dlpack(A).mark_layout_dynamic()
        b_cute = from_dlpack(B).mark_layout_dynamic()
        c_cute = from_dlpack(C).mark_layout_dynamic()
        _compiled_fn = cute.compile(jit_fn, a_cute, b_cute, c_cute)
    _compiled_fn(from_dlpack(A), from_dlpack(B), from_dlpack(C))
```

---

## Complete Examples

### Python Solution

```json
{
  "name": "swiglu_python_v1",
  "definition": "gemma3...swiglu_h4096",
  "author": "my-agent",
  "spec": {
    "languages": ["pytorch"],
    "target_hardware": ["B200"],
    "entry_point": "kernel.py::run",
    "dependencies": ["torch"],
    "destination_passing_style": true
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import torch\n\ndef run(gate, up, output):\n    output[:] = torch.nn.functional.silu(gate) * up"
    }
  ]
}
```

### Triton Solution

```json
{
  "name": "rmsnorm_triton_v1",
  "definition": "nvidia...rmsnorm_h4096",
  "author": "my-agent",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["B200"],
    "entry_point": "kernel.py::run",
    "dependencies": ["torch", "triton >= 2.3"],
    "destination_passing_style": true
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef _rmsnorm_kernel(...):\n    ...\n\ndef run(input, weight, eps, output):\n    n_rows, n_cols = input.shape\n    grid = (n_rows,)\n    _rmsnorm_kernel[grid](input, weight, output, n_cols, eps, BLOCK_SIZE=1024)"
    }
  ]
}
```

### CUDA C++ Solution

```json
{
  "name": "rope_cuda_v1",
  "definition": "flux...rope_apply_rotation",
  "author": "my-agent",
  "spec": {
    "languages": ["cuda_cpp"],
    "target_hardware": ["B200"],
    "entry_point": "kernel.cu::run",
    "dependencies": [],
    "destination_passing_style": true,
    "compile_options": {
      "cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]
    }
  },
  "sources": [
    {
      "path": "kernel.cu",
      "content": "#include <torch/extension.h>\n\n__global__ void rope_kernel(...) { ... }\n\nvoid run(torch::Tensor input, torch::Tensor freqs, torch::Tensor output) {\n    // launch rope_kernel, write into output\n}"
    }
  ]
}
```

### CuTe DSL Solution

```json
{
  "name": "attn_proj_cutedsl_v1",
  "definition": "ai21labs...attention_output_projection",
  "author": "my-agent",
  "spec": {
    "languages": ["cute_dsl"],
    "target_hardware": ["B200"],
    "entry_point": "kernel.py::run",
    "dependencies": ["torch", "cutlass"],
    "destination_passing_style": true
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import cutlass.cute as cute\nfrom cuda.core.experimental._dlpack import from_dlpack\n\n_compiled_fn = None\n\ndef run(A, B, C):\n    global _compiled_fn\n    if _compiled_fn is None:\n        _compiled_fn = cute.compile(jit_fn, ...)\n    _compiled_fn(from_dlpack(A), from_dlpack(B), from_dlpack(C))"
    }
  ]
}
```
