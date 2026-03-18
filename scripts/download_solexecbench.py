#!/usr/bin/env python3
"""Download the nvidia/SOL-ExecBench dataset from HuggingFace and unpack it
into the local ``data/benchmark/<subset>/<problem>/`` directory layout expected
by sol-execbench.

Each problem directory contains:
  - definition.json
  - reference.py
  - workload.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

REPO_ID = "nvidia/SOL-ExecBench"
SUBSETS = ["L1", "L2", "Quant", "FlashInfer-Bench"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark"


def _build_definition(row: dict) -> dict:
    """Assemble a definition.json dict from a dataset row."""
    definition: dict = {
        "name": row["name"],
        "hf_id": row.get("hf_id", None),
        "description": row["description"],
        "axes": json.loads(row["axes"]),
        "custom_inputs_entrypoint": row.get("custom_inputs_entrypoint", None),
        "inputs": json.loads(row["inputs"]),
        "outputs": json.loads(row["outputs"]),
        "reference": row["reference"],
    }
    return definition


def _process_subset(subset: str) -> None:
    print(f"Downloading {REPO_ID} config={subset} ...")
    ds = load_dataset(REPO_ID, name=subset, split="train")

    for row in ds:
        name = row["name"]
        problem_dir = OUTPUT_DIR / subset / name
        problem_dir.mkdir(parents=True, exist_ok=True)

        # definition.json
        definition = _build_definition(row)
        (problem_dir / "definition.json").write_text(
            json.dumps(definition, indent=4) + "\n"
        )

        # reference.py
        (problem_dir / "reference.py").write_text(row["reference"])

        # workload.jsonl
        workloads = json.loads(row["workloads"])
        with open(problem_dir / "workload.jsonl", "w") as f:
            for workload in workloads:
                f.write(json.dumps(workload) + "\n")

    print(f"  -> {len(ds)} problems written to {OUTPUT_DIR / subset}")


def main() -> None:
    for subset in SUBSETS:
        _process_subset(subset)
    print("Done.")


if __name__ == "__main__":
    main()
