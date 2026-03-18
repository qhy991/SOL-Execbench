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


"""Tests for sol_execbench.core.bench.clock_lock."""

from unittest.mock import MagicMock, call, patch

import pytest

from sol_execbench.core.bench.clock_lock import (
    are_clocks_locked,
    lock_clocks,
    probe_clock_lock_available,
    unlock_clocks,
    verify_clocks,
)
from sol_execbench.core.bench.config.device_config import (
    CLOCK_LOCK_PRESETS,
    ClockPreset,
    get_clock_preset,
)

_MODULE = "sol_execbench.core.bench.clock_lock"

# ── probe_clock_lock_available ────────────────────────────────────────────────


class TestProbeClockLockAvailable:
    def test_returns_true_when_sudo_succeeds(self):
        probe_result = MagicMock(returncode=0)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
            result = probe_clock_lock_available()

        assert result is True
        # Should call lgc probe then rgc reset
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0] == call(
            ["sudo", "-n", "nvidia-smi", "-lgc", "1"], capture_output=True
        )
        assert mock_run.call_args_list[1] == call(
            ["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True
        )

    def test_returns_false_when_sudo_fails(self):
        probe_result = MagicMock(returncode=1)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
            result = probe_clock_lock_available()

        assert result is False
        assert mock_run.call_count == 1  # no reset call when probe fails

    def test_returns_false_when_sudo_not_found(self):
        with patch(
            f"{_MODULE}.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = probe_clock_lock_available()

        assert result is False


# ── lock_clocks ──────────────────────────────────────────────────────────────


class TestLockClocks:
    """Tests for lock_clocks().

    verify_clocks() and time.sleep() are mocked so we only test the
    locking logic.  Verification is tested separately in TestVerifyClocks.
    """

    @staticmethod
    @pytest.fixture(autouse=True)
    def _clean_env(monkeypatch):
        """Ensure clock env vars are unset so preset logic is exercised."""
        monkeypatch.delenv("SOL_EXECBENCH_GPU_CLK_MHZ", raising=False)
        monkeypatch.delenv("SOL_EXECBENCH_DRAM_CLK_MHZ", raising=False)

    def _patch_verify_and_sleep(self):
        """Return a combined patch context that mocks verify_clocks → True and time.sleep → noop."""
        return (
            patch(f"{_MODULE}.verify_clocks", return_value=True),
            patch(f"{_MODULE}.time.sleep"),
        )

    def test_locks_gpu_and_dram_for_known_device(self):
        """Known device → calls -lgc and -lmc with correct frequencies."""
        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("NVIDIA B200")

        assert result is True
        assert mock_run.call_args_list[0] == call(
            ["sudo", "-n", "nvidia-smi", "-lgc", "1500"],
            check=True,
            capture_output=True,
        )
        assert mock_run.call_args_list[1] == call(
            ["sudo", "-n", "nvidia-smi", "-lmc", "3996"],
            check=True,
            capture_output=True,
        )

    def test_locks_h100_at_correct_frequencies(self):
        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("NVIDIA H100 SXM5 80GB")

        assert result is True
        assert mock_run.call_args_list[0] == call(
            ["sudo", "-n", "nvidia-smi", "-lgc", "1410"],
            check=True,
            capture_output=True,
        )
        assert mock_run.call_args_list[1] == call(
            ["sudo", "-n", "nvidia-smi", "-lmc", "1593"],
            check=True,
            capture_output=True,
        )

    def test_locks_a100_at_correct_frequencies(self):
        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("NVIDIA A100-SXM4-80GB")

        assert result is True
        assert mock_run.call_args_list[0] == call(
            ["sudo", "-n", "nvidia-smi", "-lgc", "1065"],
            check=True,
            capture_output=True,
        )
        assert mock_run.call_args_list[1] == call(
            ["sudo", "-n", "nvidia-smi", "-lmc", "1215"],
            check=True,
            capture_output=True,
        )

    def test_returns_false_for_unknown_device(self):
        """Unknown device with no env var overrides → False (no subprocess calls)."""
        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("NVIDIA RTX 4090")

        assert result is False
        mock_run.assert_not_called()

    def test_env_var_overrides_preset(self, monkeypatch):
        """SOL_EXECBENCH_GPU_CLK_MHZ and SOL_EXECBENCH_DRAM_CLK_MHZ override preset values."""
        monkeypatch.setenv("SOL_EXECBENCH_GPU_CLK_MHZ", "2000")
        monkeypatch.setenv("SOL_EXECBENCH_DRAM_CLK_MHZ", "4000")

        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("NVIDIA B200")

        assert result is True
        assert mock_run.call_args_list[0] == call(
            ["sudo", "-n", "nvidia-smi", "-lgc", "2000"],
            check=True,
            capture_output=True,
        )
        assert mock_run.call_args_list[1] == call(
            ["sudo", "-n", "nvidia-smi", "-lmc", "4000"],
            check=True,
            capture_output=True,
        )

    def test_env_var_overrides_unknown_device(self, monkeypatch):
        """Env var overrides allow locking even for unknown devices."""
        monkeypatch.setenv("SOL_EXECBENCH_GPU_CLK_MHZ", "1800")
        monkeypatch.setenv("SOL_EXECBENCH_DRAM_CLK_MHZ", "3500")

        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("NVIDIA UnknownGPU9999")

        assert result is True
        assert mock_run.call_args_list[0] == call(
            ["sudo", "-n", "nvidia-smi", "-lgc", "1800"],
            check=True,
            capture_output=True,
        )
        assert mock_run.call_args_list[1] == call(
            ["sudo", "-n", "nvidia-smi", "-lmc", "3500"],
            check=True,
            capture_output=True,
        )

    def test_returns_false_when_lgc_fails(self):
        """GPU clock lock failure → returns False, no DRAM lock attempt."""
        import subprocess as sp

        with patch(
            f"{_MODULE}.subprocess.run",
            side_effect=sp.CalledProcessError(1, "nvidia-smi"),
        ) as mock_run:
            result = lock_clocks("NVIDIA B200")

        assert result is False
        assert mock_run.call_count == 1  # only lgc, no lmc

    def test_returns_false_and_unlocks_gpu_when_lmc_fails(self):
        """DRAM lock failure → unlocks GPU clocks and returns False."""
        import subprocess as sp

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock()  # lgc succeeds
            elif call_count[0] == 2:
                raise sp.CalledProcessError(1, "nvidia-smi")  # lmc fails
            return MagicMock()  # rgc cleanup

        with patch(f"{_MODULE}.subprocess.run", side_effect=side_effect) as mock_run:
            result = lock_clocks("NVIDIA B200")

        assert result is False
        # 3 calls: lgc, lmc (fails), rgc (cleanup)
        assert mock_run.call_count == 3

    def test_returns_false_when_verification_fails(self):
        """Successful lgc+lmc but verify_clocks fails → unlock and return False."""
        p_sleep = patch(f"{_MODULE}.time.sleep")
        p_verify = patch(f"{_MODULE}.verify_clocks", return_value=False)
        with (
            patch(f"{_MODULE}.subprocess.run") as mock_run,
            p_verify,
            p_sleep,
        ):
            result = lock_clocks("NVIDIA B200")

        assert result is False
        # lgc, lmc, then unlock (rgc + rmc)
        assert mock_run.call_count == 4

    def test_sleeps_before_verification(self):
        """lock_clocks() sleeps VERIFY_DELAY_S before calling verify_clocks."""
        from sol_execbench.core.bench.clock_lock import VERIFY_DELAY_S

        p_verify = patch(f"{_MODULE}.verify_clocks", return_value=True)
        p_sleep = patch(f"{_MODULE}.time.sleep")
        with (
            patch(f"{_MODULE}.subprocess.run"),
            p_verify,
            p_sleep as mock_sleep,
        ):
            lock_clocks("NVIDIA B200")

        mock_sleep.assert_called_once_with(VERIFY_DELAY_S)


# ── verify_clocks ───────────────────────────────────────────────────────────


class TestVerifyClocks:
    def _make_smi_result(self, stdout: str, returncode: int = 0):
        return MagicMock(returncode=returncode, stdout=stdout, stderr="")

    def test_single_gpu_within_tolerance(self):
        result = self._make_smi_result("1500, 3996\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1500, 3996) is True

    def test_single_gpu_within_tolerance_margin(self):
        """Clock 30 MHz off but within default 50 MHz tolerance."""
        result = self._make_smi_result("1530, 3970\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1500, 3996) is True

    def test_single_gpu_outside_tolerance(self):
        """GPU clock 100 MHz off → fails."""
        result = self._make_smi_result("1400, 3996\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1500, 3996) is False

    def test_dram_outside_tolerance(self):
        result = self._make_smi_result("1500, 3800\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1500, 3996) is False

    def test_multi_gpu_all_match(self):
        result = self._make_smi_result("1410, 1593\n1410, 1593\n1410, 1593\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1410, 1593) is True

    def test_multi_gpu_one_mismatch(self):
        result = self._make_smi_result("1410, 1593\n1410, 1593\n1200, 1593\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1410, 1593) is False

    def test_nvidia_smi_not_found(self):
        with patch(f"{_MODULE}.subprocess.run", side_effect=FileNotFoundError):
            assert verify_clocks(1500, 3996) is False

    def test_nvidia_smi_nonzero_exit(self):
        result = self._make_smi_result("", returncode=1)
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1500, 3996) is False

    def test_empty_output(self):
        result = self._make_smi_result("")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1500, 3996) is False

    def test_malformed_output(self):
        result = self._make_smi_result("not_a_number, also_not\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1500, 3996) is False

    def test_custom_tolerance(self):
        """tolerance_mhz=10 makes a 30 MHz deviation fail."""
        result = self._make_smi_result("1530, 3996\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1500, 3996, tolerance_mhz=10) is False


# ── unlock_clocks ────────────────────────────────────────────────────────────


class TestUnlockClocks:
    def test_calls_rgc_and_rmc(self):
        with patch(f"{_MODULE}.subprocess.run") as mock_run:
            unlock_clocks()

        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0] == call(
            ["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True
        )
        assert mock_run.call_args_list[1] == call(
            ["sudo", "-n", "nvidia-smi", "-rmc"], capture_output=True
        )

    def test_does_not_raise_on_failure(self):
        """unlock_clocks is best-effort — exceptions are swallowed."""
        with patch(
            f"{_MODULE}.subprocess.run",
            side_effect=Exception("no sudo"),
        ):
            unlock_clocks()  # should not raise


# ── are_clocks_locked ────────────────────────────────────────────────────────


class TestAreClocksLocked:
    def test_returns_true_when_env_set(self, monkeypatch):
        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "1")
        assert are_clocks_locked() is True

    def test_returns_false_when_env_zero(self, monkeypatch):
        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "0")
        assert are_clocks_locked() is False

    def test_returns_false_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SOL_EXECBENCH_CLOCKS_LOCKED", raising=False)
        assert are_clocks_locked() is False


# ── get_clock_preset / CLOCK_LOCK_PRESETS ────────────────────────────────────


class TestGetClockPreset:
    def test_b200(self):
        preset = get_clock_preset("NVIDIA B200")
        assert preset == ClockPreset(gpu_clk_mhz=1500, dram_clk_mhz=3996)

    def test_h100_exact(self):
        preset = get_clock_preset("NVIDIA H100")
        assert preset == ClockPreset(gpu_clk_mhz=1410, dram_clk_mhz=1593)

    def test_h100_with_full_product_name(self):
        preset = get_clock_preset("NVIDIA H100 SXM5 80GB HBM3")
        assert preset is not None
        assert preset.gpu_clk_mhz == 1410
        assert preset.dram_clk_mhz == 1593

    def test_a100_with_full_product_name(self):
        preset = get_clock_preset("NVIDIA A100-SXM4-80GB")
        assert preset is not None
        assert preset.gpu_clk_mhz == 1065
        assert preset.dram_clk_mhz == 1215

    def test_unknown_device_returns_none(self):
        assert get_clock_preset("NVIDIA RTX 4090") is None

    def test_empty_string_returns_none(self):
        assert get_clock_preset("") is None

    def test_presets_cover_all_known_devices(self):
        """Each preset key resolves to its own preset."""
        for name, preset in CLOCK_LOCK_PRESETS.items():
            assert get_clock_preset(name) == preset
