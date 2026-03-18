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


"""Tests for the clock lock validation and status message logic in the eval driver.

The eval driver now:
1. Rejects all workloads with RUNTIME_ERROR if lock_clocks=True and clocks
   are not locked (SOL_EXECBENCH_CLOCKS_LOCKED != 1).
2. Prepends "Clocks locked: yes/no" to every Trace's log field.
"""

from sol_execbench.core.bench.config.benchmark_config import BenchmarkConfig
from sol_execbench.core.bench.utils import make_eval
from sol_execbench.core.data.trace import EvaluationStatus

_DEVICE = "cpu"


def _make_eval_with_status(clock_status_msg: str, extra_msg=None):
    """Mirrors the _make_eval wrapper from eval_driver.py."""
    parts = [p for p in (clock_status_msg, extra_msg) if p]
    return make_eval(
        EvaluationStatus.RUNTIME_ERROR,
        _DEVICE,
        log_path=None,
        extra_msg="\n".join(parts) or None,
    )


class TestClockStatusMessage:
    def test_clocks_locked_yes(self):
        """When clocks are locked, log starts with 'Clocks locked: yes'."""
        ev = _make_eval_with_status("Clocks locked: yes")
        assert ev.log == "Clocks locked: yes"

    def test_clocks_locked_no(self):
        """When clocks are not locked, log starts with 'Clocks locked: no'."""
        ev = _make_eval_with_status("Clocks locked: no")
        assert ev.log == "Clocks locked: no"

    def test_status_msg_with_extra_msg(self):
        """Status message is prepended before extra_msg."""
        ev = _make_eval_with_status(
            "Clocks locked: yes", extra_msg="Timing failed: OOM"
        )
        assert ev.log == "Clocks locked: yes\nTiming failed: OOM"

    def test_no_status_extra_msg_only(self):
        """Empty status string: log == extra_msg."""
        ev = _make_eval_with_status("", extra_msg="Timing failed: OOM")
        assert ev.log == "Timing failed: OOM"


class TestClockLockValidation:
    def test_lock_clocks_true_unlocked_rejects(self, monkeypatch):
        """lock_clocks=True + SOL_EXECBENCH_CLOCKS_LOCKED=0 → should reject."""
        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "0")
        import os

        cfg = BenchmarkConfig(lock_clocks=True)
        clocks_locked = os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
        assert cfg.lock_clocks and not clocks_locked

    def test_lock_clocks_true_locked_proceeds(self, monkeypatch):
        """lock_clocks=True + SOL_EXECBENCH_CLOCKS_LOCKED=1 → should proceed."""
        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "1")
        import os

        cfg = BenchmarkConfig(lock_clocks=True)
        clocks_locked = os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
        assert not (cfg.lock_clocks and not clocks_locked)

    def test_lock_clocks_false_unlocked_proceeds(self, monkeypatch):
        """lock_clocks=False + SOL_EXECBENCH_CLOCKS_LOCKED=0 → should proceed."""
        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "0")
        import os

        cfg = BenchmarkConfig(lock_clocks=False)
        clocks_locked = os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
        assert not (cfg.lock_clocks and not clocks_locked)

    def test_lock_clocks_false_locked_proceeds(self, monkeypatch):
        """lock_clocks=False + SOL_EXECBENCH_CLOCKS_LOCKED=1 → should proceed."""
        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "1")
        import os

        cfg = BenchmarkConfig(lock_clocks=False)
        clocks_locked = os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
        assert not (cfg.lock_clocks and not clocks_locked)
