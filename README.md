# SOL ExecBench

<div align="center" id="top">

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](/docs/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[HuggingFace Dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench) | [Leaderboard](https://research.nvidia.com/benchmarks/sol-execbench)
| [Arxiv (coming soon)]() </div>

**Speed-Of-Light ExecBench** is a rigorous GPU kernel evaluation and benchmarking framework built to benchmark AI-generated kernel solutions written with the variety of DSLs that NVIDIA hardware supports.

Kernels are:
* Checked for various forms of reward hacking
* Tested against a reference solution for numerical correctness
* Timed under reproducible conditions

Leaderboard submissions are ranked based on [SOL-Score](/src/sol_execbench/sol_score.py): a metric that grades custom kernel performance based on the theoretical roofline of a NVIDIA B200 GPU (obtained analytically with [SOLAR](https://github.com/NVlabs/SOLAR)).  

Supported kernel languages: PyTorch, Triton, CUTLASS, cuDNN, CuTe DSL, cuTile, CUDA C++.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install huggingface-hub[cli]`)
- NVIDIA driver version 580+

## Setup

### 1. Download benchmark data (one-time)

```bash
./scripts/download_data.sh
```

This downloads the [SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench) and [FlashInfer Trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) datasets into `data/`.

### 2. Build and launch the Docker container

```bash
./scripts/run_docker.sh --build
```

This builds the image and drops you into an interactive shell inside the container. The repo's `src/`, `tests/`, and downloaded data are mounted automatically.

## Evaluating a Solution

Inside the container, use the `sol-execbench` CLI:

```bash
# Evaluate using a problem directory (contains definition.json + workload.jsonl)
sol-execbench <problem_dir> --solution solution.json

# Or specify files explicitly
sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json
```

### Example

```bash
# From the host — build, launch, and evaluate in one command:
./scripts/run_docker.sh --build -- \
  sol-execbench examples/cute_dsl/jamba_attn_proj \
    --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json

# Or from inside the container:
sol-execbench examples/cute_dsl/jamba_attn_proj \
  --solution examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json
```

### CLI Options

| Flag | Description |
|---|---|
| `--compile-timeout` | Compilation timeout in seconds (default: 120) |
| `--timeout` | Evaluation timeout in seconds (default: 600) |
| `-o, --output` | Write JSONL traces to file |
| `--json` | Print traces as JSON to stdout |
| `--lock-clocks` | Lock GPU clocks for stable benchmarks |
| `--keep-staging` | Preserve staging directory after run |
| `-v, --verbose` | Show subprocess output |

## Running a Dataset

Use `scripts/run_dataset.py` to evaluate an entire dataset (or a single problem) in batch. By default it runs the definition's reference implementation as the solution unless `--solution-name` is specified. Saves to `./out/{subset}` by default.

```bash
# Run all problems in the benchmark.
# Auto builds solution.json from a single code file
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --solution-name solution.py

# Run specific categories with multiple solution code files
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --category L1 L2 --solution-name solution.json

# Run a single problem
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark/L1/my_problem

# Limit number of problems and workloads
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5 --max-workloads 3 -o ./results
```

Results (traces and a summary JSON) are written to `out/run_dataset/` by default (override with `-o`). Problems that already passed are skipped on subsequent runs unless `--rerun` is specified.

## Problem Format

A problem directory contains:

- **`definition.json`** — Kernel specification: function signature, tensor shapes, dtypes, reference implementation.
- **`workload.jsonl`** — One JSON object per line, each defining input shapes, values, and tolerance thresholds.

A solution is a separate JSON file referencing source files with the kernel implementation.

See the full schema docs:

- [Definition](docs/definition.md) — Kernel specification (function signature, tensor shapes, dtypes, reference code)
- [Workload](docs/workload.md) — Concrete input configurations and tolerance thresholds
- [Solution](docs/solution.md) — Source files and build specs for a kernel implementation
- [Trace](docs/trace.md) — Evaluation output (correctness and performance results)

## License

Apache-2.0. See [LICENSE](LICENSE). Contributions require DCO sign-off — see [CONTRIBUTING.md](CONTRIBUTING.md).
