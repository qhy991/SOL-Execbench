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

"""Tests for the SOL reference table loader and SOL-ExecBench Eq. (2) score.

These tests do not require a GPU. They use a tiny synthetic CSV written to a
temp file and point the loader at it via the ``SOL_TSOL_TBASE_CSV`` env var.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sol_execbench.core import sol_reference as ref_mod
from sol_execbench.core.sol_reference import (
    paper_sol_score,
    lookup_sol_ref,
    sol_score_for,
    load_sol_reference,
)


@pytest.fixture
def sol_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a 3-row synthetic SOL reference CSV and point the loader at it."""
    csv = tmp_path / "tsol_tbase.csv"
    csv.write_text(
        "category,kernel,workload_index,workload_uuid,t_sol_ms,t_base_ms\n"
        "L2,009_decoder_layer,0,uuid-A,1.0,3.0\n"
        "L2,009_decoder_layer,1,uuid-B,0.5,2.0\n"
        "L2,009_decoder_layer,2,uuid-C,2.0,8.0\n"
    )
    monkeypatch.setenv("SOL_TSOL_TBASE_CSV", str(csv))
    ref_mod.load_sol_reference.cache_clear()
    yield csv
    ref_mod.load_sol_reference.cache_clear()


def test_paper_sol_score_boundaries():
    # S=1 at T_k=T_sol, S=0.5 at T_k=T_base
    assert paper_sol_score(1.0, 1.0, 3.0) == pytest.approx(1.0)
    assert paper_sol_score(3.0, 1.0, 3.0) == pytest.approx(0.5)
    # half-way -> 2/3
    assert paper_sol_score(2.0, 1.0, 3.0) == pytest.approx(2 / 3)
    # saturates at 1.0 below T_sol
    assert paper_sol_score(0.5, 1.0, 3.0) == pytest.approx(1.0)
    # very large latency -> score approaches 0 (asymptotic, never reaches 0)
    assert paper_sol_score(100.0, 1.0, 3.0) == pytest.approx(0.0198, abs=1e-4)


def test_paper_sol_score_invalid():
    assert paper_sol_score(None, 1.0, 3.0) is None
    assert paper_sol_score(1.0, None, 3.0) is None
    assert paper_sol_score(1.0, 1.0, None) is None
    assert paper_sol_score(0.0, 1.0, 3.0) is None
    assert paper_sol_score(-1.0, 1.0, 3.0) is None
    # gap <= 0 (t_base <= t_sol) is undefined
    assert paper_sol_score(1.0, 3.0, 3.0) is None
    assert paper_sol_score(1.0, 3.0, 1.0) is None


def test_load_sol_reference(sol_csv):
    table = load_sol_reference()
    assert table is not None
    assert len(table) == 3
    assert "uuid-A" in table
    assert table["uuid-A"].t_sol_ms == 1.0
    assert table["uuid-A"].t_base_ms == 3.0


def test_lookup_sol_ref(sol_csv):
    ref = lookup_sol_ref("uuid-B")
    assert ref is not None
    assert ref.t_sol_ms == 0.5
    assert ref.t_base_ms == 2.0
    # unknown uuid -> None
    assert lookup_sol_ref("not-a-uuid") is None


def test_sol_score_for_contest_workload(sol_csv):
    # uuid-A: t_sol=1, t_base=3 -> at latency 1.0 score is 1.0
    assert sol_score_for("uuid-A", 1.0) == pytest.approx(1.0)
    # at latency 3.0 score is 0.5
    assert sol_score_for("uuid-A", 3.0) == pytest.approx(0.5)


def test_sol_score_for_non_contest_returns_none(sol_csv):
    """Non-contest workloads (uuid not in table) get no SOL score."""
    assert sol_score_for("not-a-uuid", 1.0) is None


def test_no_csv_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """When no CSV can be resolved, the loader returns None and lookups no-op."""
    monkeypatch.delenv("SOL_TSOL_TBASE_CSV", raising=False)
    monkeypatch.delenv("SOLEXECBENCH_SOL_CSV", raising=False)
    monkeypatch.chdir(tmp_path)
    ref_mod.load_sol_reference.cache_clear()
    try:
        assert load_sol_reference() is None
        assert lookup_sol_ref("any-uuid") is None
        assert sol_score_for("any-uuid", 1.0) is None
    finally:
        ref_mod.load_sol_reference.cache_clear()
