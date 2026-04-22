# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL-ExecBench CLI — evaluate solutions locally on GPU.

Usage:
    uv run sol-execbench <problem_dir> --solution solution.json
    uv run sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json

By default, traces are printed as one JSON object per line on stdout. Pass --no-json for a human-readable Rich table on stderr (success path only).

If compilation or evaluation fails before traces exist, **one** JSON object is **always** written to stdout (and to ``-o`` when given) with ``parse_status`` set—this does not depend on ``--json`` / ``--no-json``, so callers always receive machine-readable errors on stdout.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core import (
    Definition,
    Workload,
    Solution,
    Trace,
    BenchmarkConfig,
    EvaluationStatus,
)
from ..driver import ProblemPackager

console = Console(stderr=True)

# Cap embedded compiler logs in JSON error payloads (bytes-ish; UTF-8 safe slice may shorten).
_MAX_DIAG_CHARS = 24000


def _truncate_diag(text: str, max_chars: int = _MAX_DIAG_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _extract_compiler_error_summary(text: str, max_lines: int = 32) -> str:
    """Extract actionable compiler error lines from noisy build logs.

    nvcc/g++/ninja errors are usually buried in the middle of the log.
    This pulls the most diagnostic-critical lines (error:, fatal error,
    undefined reference, etc.) so agents don't have to scroll through
    thousands of lines of context.
    """
    if not text:
        return ""
    picked: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        lower = s.lower()
        hit = False
        if "error:" in lower or "fatal error" in lower:
            hit = True
        elif "undefined reference" in lower:
            hit = True
        elif "collect2:" in lower and "error" in lower:
            hit = True
        elif "nvcc fatal" in lower:
            hit = True
        elif lower.startswith("ninja:") and ("error" in lower or "failed" in lower):
            hit = True
        elif "failed:" in lower and any(ext in lower for ext in (".cu", ".cpp", ".cc", ".cuh")):
            hit = True
        elif "note:" in lower and ("candidate" in lower or "declared" in lower):
            hit = True
        if not hit:
            continue
        clipped = s[:400]
        if clipped not in seen:
            seen.add(clipped)
            picked.append(clipped)
        if len(picked) >= max_lines:
            break
    return "\n".join(picked)


def _emit_json_failure(
    *,
    parse_status: str,
    phase: str,
    problem: str,
    solution: str,
    output_file: Optional[Path],
    fields: dict[str, Any],
) -> None:
    """Emit a single JSON object to stdout (and optionally -o) for agent/CI consumption.

    Always prints to stdout so failure diagnostics are available even when ``--no-json``
    is used for the success path.
    """
    payload: dict[str, Any] = {
        "parse_status": parse_status,
        "phase": phase,
        "problem": problem,
        "solution": solution,
        **fields,
    }
    line = json.dumps(payload, ensure_ascii=False)
    print(line, file=sys.stdout, flush=True)
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(line + "\n", encoding="utf-8")


def _normalize_subprocess_command(cmd: list[str]) -> list[str]:
    normalized = [str(part) for part in list(cmd or [])]
    if normalized and normalized[0].strip() == "python":
        normalized[0] = sys.executable
    return normalized


def _load_definition(path: Path) -> Definition:
    return Definition(**json.loads(path.read_text()))


def _load_workloads(path: Path) -> list[Workload]:
    workloads = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            workloads.append(Workload(**json.loads(line)))
    return workloads


def _load_solution(path: Path) -> Solution:
    sol_dict = json.loads(path.read_text())
    # Resolve source file contents relative to the solution JSON directory.
    sol_dir = path.parent
    for src in sol_dict.get("sources", []):
        if not src.get("content"):
            src_path = sol_dir / src["path"]
            if src_path.exists():
                src["content"] = src_path.read_text()
    return Solution(**sol_dict)


def _load_config(path: Optional[Path]) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    return BenchmarkConfig(**json.loads(path.read_text()))


def _resolve_problem_dir(
    problem_dir: Path,
) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
    """Return (definition.json, workload.jsonl, config.json?, solution.json?) inside a problem directory."""
    def_path = problem_dir / "definition.json"
    wkl_path = problem_dir / "workload.jsonl"
    cfg_path = problem_dir / "config.json"
    sol_path = problem_dir / "solution.json"
    if not def_path.exists():
        raise click.ClickException(f"definition.json not found in {problem_dir}")
    if not wkl_path.exists():
        raise click.ClickException(f"workload.jsonl not found in {problem_dir}")
    return (
        def_path,
        wkl_path,
        cfg_path if cfg_path.exists() else None,
        sol_path if sol_path.exists() else None,
    )


def _print_traces_table(traces: list[Trace]) -> None:
    """Print a rich table summarizing evaluation traces."""

    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Status", width=22)
    table.add_column("Latency (ms)", justify="right", width=14)
    table.add_column("Ref (ms)", justify="right", width=14)
    table.add_column("Speedup", justify="right", width=10)
    table.add_column("Max Abs Err", justify="right", width=14)
    table.add_column("Max Rel Err", justify="right", width=14)

    passed = 0
    total = len(traces)
    for i, trace in enumerate(traces):
        ev = trace.evaluation
        if ev is None:
            table.add_row(str(i), "[dim]no evaluation[/dim]", "", "", "", "", "")
            continue

        status = ev.status.value
        if ev.status == EvaluationStatus.PASSED:
            status_str = f"[green]{status}[/green]"
            passed += 1
        elif ev.status == EvaluationStatus.INCORRECT_NUMERICAL:
            status_str = f"[yellow]{status}[/yellow]"
        else:
            status_str = f"[red]{status}[/red]"

        latency = ""
        ref_latency = ""
        speedup = ""
        if ev.performance:
            latency = f"{ev.performance.latency_ms:.3f}"
            ref_latency = f"{ev.performance.reference_latency_ms:.3f}"
            speedup = f"{ev.performance.speedup_factor:.2f}x"

        abs_err = ""
        rel_err = ""
        if ev.correctness:
            if ev.correctness.has_nan:
                abs_err = "NaN"
                rel_err = "NaN"
            elif ev.correctness.has_inf:
                abs_err = "Inf"
                rel_err = "Inf"
            else:
                abs_err = f"{ev.correctness.max_absolute_error:.2e}"
                rel_err = f"{ev.correctness.max_relative_error:.2e}"

        table.add_row(
            str(i), status_str, latency, ref_latency, speedup, abs_err, rel_err
        )

    console.print(table)
    console.print(f"\n[bold]{passed}/{total}[/bold] workloads passed")

    # Show logs for traces with runtime errors
    error_logs = []
    for i, trace in enumerate(traces):
        ev = trace.evaluation
        if ev is None:
            continue
        if (
            ev.status
            not in (EvaluationStatus.PASSED, EvaluationStatus.INCORRECT_NUMERICAL)
            and ev.log
        ):
            error_logs.append((i, ev.status.value, ev.log))

    if error_logs:
        console.print(f"\n[bold red]Runtime logs ({len(error_logs)}):[/bold red]")
        for idx, status, log in error_logs:
            console.print(f"\n[bold]Workload {idx}[/bold] ([red]{status}[/red]):")
            console.print(log.rstrip())


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "problem_dir", required=False, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--definition",
    "definition_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to definition.json",
)
@click.option(
    "--workload",
    "workload_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to workload.jsonl",
)
@click.option(
    "--solution",
    "solution_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to solution.json",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to benchmark config JSON",
)
@click.option(
    "--compile-timeout",
    default=120,
    type=int,
    help="Compilation timeout in seconds (C++/CUDA only)",
)
@click.option(
    "--timeout", default=600, type=int, help="Evaluation subprocess timeout in seconds"
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(path_type=Path),
    help="Write trace JSONL to this file",
)
@click.option(
    "--json/--no-json",
    "json_output",
    default=True,
    help="On success: print one JSON trace per line to stdout (default); --no-json prints a Rich table on stderr. On failure: one JSON error object is always printed to stdout (and to -o if set), regardless of this flag.",
)
@click.option("--lock-clocks", is_flag=True, help="Require GPU clocks to be locked")
@click.option(
    "--keep-staging", is_flag=True, help="Keep the staging directory after evaluation"
)
@click.option("--verbose", "-v", is_flag=True, help="Show subprocess output")
def cli(
    problem_dir: Optional[Path],
    definition_file: Optional[Path],
    workload_file: Optional[Path],
    solution_file: Path,
    config_file: Optional[Path],
    compile_timeout: int,
    timeout: int,
    output_file: Optional[Path],
    json_output: bool,
    lock_clocks: bool,
    keep_staging: bool,
    verbose: bool,
):
    """Evaluate a SOL-ExecBench solution on GPU.

    \b
    Two ways to specify the problem:
      1) Positional: sol-execbench <problem_dir> --solution sol.json
         (reads definition.json and workload.jsonl from problem_dir)
      2) Explicit:   sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json
    """
    # Resolve definition + workloads
    if problem_dir:
        def_path, wkl_path, cfg_path, sol_path = _resolve_problem_dir(problem_dir)
        definition_file = definition_file or def_path
        workload_file = workload_file or wkl_path
        config_file = config_file or cfg_path
        solution_file = solution_file or sol_path

    if not definition_file:
        raise click.ClickException("Provide PROBLEM_DIR or --definition")
    if not workload_file:
        raise click.ClickException("Provide PROBLEM_DIR or --workload")
    if not solution_file:
        raise click.ClickException(
            "Provide PROBLEM_DIR with solution.json or --solution"
        )

    # Load data models
    definition = _load_definition(definition_file)
    workloads = _load_workloads(workload_file)
    solution = _load_solution(solution_file)
    config = _load_config(config_file)

    if lock_clocks:
        config.lock_clocks = True

    console.print(f"[bold]Problem:[/bold]  {definition.name}")
    console.print(f"[bold]Solution:[/bold] {solution.name}")
    console.print(f"[bold]Workloads:[/bold] {len(workloads)}")
    if config_file:
        console.print(
            f"[bold]Config:[/bold]   {json.dumps(dataclasses.asdict(config))}"
        )

    # Create staging directory
    staging_dir = Path(tempfile.mkdtemp(prefix="sol_execbench_"))
    packager = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config,
        output_dir=staging_dir,
        keep_output_dir=keep_staging,
    )

    if verbose:
        console.print(f"[dim]Staging dir: {staging_dir}[/dim]")

    # Phase 1: Compile (C++/CUDA only)
    if packager._is_cpp:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Compiling C++/CUDA solution...", total=None)

            cmd, artifact_path = packager.compile()
            cmd = _normalize_subprocess_command(cmd)
            proc = subprocess.run(
                cmd,
                cwd=staging_dir,
                capture_output=True,
                text=True,
                timeout=compile_timeout,
                env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            )
            progress.update(task, completed=True)

        if proc.returncode != 0:
            console.print("[red]Compilation failed[/red]")
            if proc.stderr:
                console.print(proc.stderr)
            if proc.stdout:
                console.print(proc.stdout)
            _emit_json_failure(
                parse_status="compilation_failed",
                phase="compile",
                problem=definition.name,
                solution=solution.name,
                output_file=output_file,
                fields={
                    "compiler_returncode": proc.returncode,
                    "compiler_stderr": _truncate_diag(proc.stderr or ""),
                    "compiler_stdout": _truncate_diag(proc.stdout or ""),
                    "compiler_error_summary": _extract_compiler_error_summary(
                        (proc.stderr or "") + "\n" + (proc.stdout or "")
                    ),
                    "hint": "C++/CUDA extension build failed before evaluation. See compiler_error_summary for actionable lines, or compiler_stderr/compiler_stdout for full logs.",
                },
            )
            sys.exit(1)

        console.print("[green]Compilation succeeded[/green]")
        if verbose and proc.stderr:
            console.print(f"[dim]{proc.stderr}[/dim]")

    # Phase 2: Evaluate
    eval_cmd = _normalize_subprocess_command(packager.execute())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Evaluating {len(workloads)} workload(s)...", total=None
        )

        proc = subprocess.run(
            eval_cmd,
            cwd=staging_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        )
        progress.update(task, completed=True)

    if verbose and proc.stderr:
        console.print(f"[dim]{proc.stderr}[/dim]")

    if proc.returncode != 0 and not proc.stdout.strip():
        console.print("[red]Evaluation failed[/red]")
        if proc.stderr:
            console.print(proc.stderr)
        _emit_json_failure(
            parse_status="evaluation_failed",
            phase="evaluate",
            problem=definition.name,
            solution=solution.name,
            output_file=output_file,
            fields={
                "subprocess_returncode": proc.returncode,
                "eval_stderr": _truncate_diag(proc.stderr or ""),
                "hint": "Evaluation subprocess exited before emitting JSONL traces on stdout.",
            },
        )
        sys.exit(1)

    # Parse traces from stdout
    traces = packager.convert_stdout_to_traces(proc.stdout)

    if not traces:
        console.print("[red]No traces produced[/red]")
        if proc.stderr:
            console.print(proc.stderr)
        _emit_json_failure(
            parse_status="no_traces",
            phase="evaluate",
            problem=definition.name,
            solution=solution.name,
            output_file=output_file,
            fields={
                "subprocess_returncode": proc.returncode,
                "eval_stderr": _truncate_diag(proc.stderr or ""),
                "eval_stdout_head": _truncate_diag(proc.stdout or ""),
                "hint": "Could not parse evaluation stdout into Trace JSONL lines.",
            },
        )
        sys.exit(1)

    # Output
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for t in traces:
                f.write(json.dumps(t.model_dump(mode="json")) + "\n")
        console.print(f"[green]Saved {len(traces)} traces to {output_file}[/green]")

    if json_output:
        for t in traces:
            print(json.dumps(t.model_dump(mode="json")))
    else:
        _print_traces_table(traces)

    # Exit code: 0 if all passed, 1 otherwise
    all_passed = all(
        t.evaluation and t.evaluation.status == EvaluationStatus.PASSED for t in traces
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    cli()
