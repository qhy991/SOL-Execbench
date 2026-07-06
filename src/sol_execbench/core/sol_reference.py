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

"""Speed-of-Light reference table for contest workloads.

Loads the precomputed (t_sol_ms, t_base_ms) table that the SoL-Contest
organizers publish per workload_uuid. The harness uses it to compute the
SOL score for contest tasks; non-contest workloads simply have no entry
and get no SOL score.

Path resolution order (first match wins):
  1. ``SOL_TSOL_TBASE_CSV`` env var (absolute path to the CSV).
  2. ``SOLEXECBENCH_SOL_CSV`` env var (alternative name).
  3. ``data/ray234/h800_final_two_number_20260702/final_tsol_tbase.csv``
     under the current repo root, cwd, or any ancestor of cwd.
  4. The same relative path under ``../SoL-Contest-InfiniAI`` (the
     sibling contest repo layout used in our environment).

The CSV is expected to have a header row with at minimum the columns
``workload_uuid``, ``t_sol_ms``, ``t_base_ms``. Extra columns are ignored.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

DEFAULT_CSV_REL = Path(
    "data/ray234/h800_final_two_number_20260702/final_tsol_tbase.csv"
)
SIBLING_CONTEST_REL = Path("../SoL-Contest-InfiniAI") / DEFAULT_CSV_REL


def paper_sol_score(
    latency_ms: Optional[float],
    t_sol_ms: Optional[float],
    t_base_ms: Optional[float],
) -> Optional[float]:
    """SOL-ExecBench Eq. (2): S in [0, 1].

    S = 1 when T_k = T_sol (speed-of-light), S = 0.5 when T_k = T_base
    (baseline reference implementation). Returns None if any input is
    missing or non-positive, or if t_base <= t_sol (no positive gap).
    """
    if latency_ms is None or t_sol_ms is None or t_base_ms is None:
        return None
    if latency_ms <= 0 or t_sol_ms <= 0 or t_base_ms <= 0:
        return None
    gap = t_base_ms - t_sol_ms
    if gap <= 0:
        return None
    score = 1.0 / (1.0 + (latency_ms - t_sol_ms) / gap)
    if score < 0:
        return 0.0
    return min(score, 1.0)


@dataclass(frozen=True)
class SolWorkloadRef:
    workload_uuid: str
    t_sol_ms: float
    t_base_ms: Optional[float] = None


def _as_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _resolve_csv_path() -> Optional[Path]:
    env = os.environ.get("SOL_TSOL_TBASE_CSV") or os.environ.get(
        "SOLEXECBENCH_SOL_CSV"
    )
    if env:
        p = Path(env)
        return p if p.is_file() else None

    # 3. search cwd and ancestors
    cwd = Path.cwd()
    for base in [cwd, *cwd.parents]:
        cand = base / DEFAULT_CSV_REL
        if cand.is_file():
            return cand

    # 4. sibling contest repo layout
    for base in [cwd, *cwd.parents]:
        cand = base / SIBLING_CONTEST_REL
        if cand.is_file():
            return cand
    return None


def _load_csv(path: Path) -> dict[str, SolWorkloadRef]:
    out: dict[str, SolWorkloadRef] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uuid = row.get("workload_uuid")
            t_sol = _as_float(row.get("t_sol_ms"))
            if not uuid or t_sol is None:
                continue
            out[str(uuid)] = SolWorkloadRef(
                workload_uuid=str(uuid),
                t_sol_ms=t_sol,
                t_base_ms=_as_float(row.get("t_base_ms")),
            )
    return out


@lru_cache(maxsize=1)
def load_sol_reference() -> Optional[dict[str, SolWorkloadRef]]:
    """Return the uuid->SolWorkloadRef map, or None if no CSV is found.

    The result is cached for the process lifetime; tests that swap the CSV
    via the env var should call ``_load_csv.cache_clear`` /
    ``load_sol_reference.cache_clear`` after changing the environment.
    """
    path = _resolve_csv_path()
    if path is None:
        return None
    return _load_csv(path)


def lookup_sol_ref(workload_uuid: str) -> Optional[SolWorkloadRef]:
    """Look up the SOL reference for a single workload_uuid."""
    table = load_sol_reference()
    if table is None:
        return None
    return table.get(workload_uuid)


def sol_score_for(workload_uuid: str, latency_ms: Optional[float]) -> Optional[float]:
    """Convenience: compute the SOL score for one workload given its latency.

    Returns None if the workload is not a contest workload (uuid not in
    the SOL reference table) or if the score cannot be computed.
    """
    ref = lookup_sol_ref(workload_uuid)
    if ref is None:
        return None
    return paper_sol_score(latency_ms, ref.t_sol_ms, ref.t_base_ms)
