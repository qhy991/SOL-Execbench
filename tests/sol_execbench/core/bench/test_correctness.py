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

import pytest
import torch

from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.data.workload import ToleranceSpec


def _spec(
    max_atol: float = 1e-5,
    max_rtol: float = 1e-5,
    required_matched_ratio: float = 1.0,
    max_error_cap: float | None = None,
    allow_negative_inf: bool = False,
) -> ToleranceSpec:
    return ToleranceSpec(
        max_atol=max_atol,
        max_rtol=max_rtol,
        required_matched_ratio=required_matched_ratio,
        max_error_cap=max_error_cap,
        allow_negative_inf=allow_negative_inf,
    )


class TestComputeErrorStats:
    """Tests for compute_error_stats, focusing on near-zero edge cases."""

    # ------------------------------------------------------------------
    # Basic / happy-path
    # ------------------------------------------------------------------

    def test_identical_tensors(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        c, exceeds = compute_error_stats(t, t, _spec())
        assert c.max_absolute_error == 0.0
        assert c.max_relative_error == 0.0
        assert not exceeds

    def test_within_tolerance(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = ref + 1e-6
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert not exceeds

    def test_exceeds_tolerance(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = ref + 1.0  # way outside tolerance
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    # ------------------------------------------------------------------
    # Empty tensor
    # ------------------------------------------------------------------

    def test_empty_tensors(self):
        t = torch.tensor([])
        c, exceeds = compute_error_stats(t, t, _spec())
        assert c.max_absolute_error == 0.0
        assert c.max_relative_error == 0.0
        assert not exceeds

    # ------------------------------------------------------------------
    # Near-zero reference values
    # ------------------------------------------------------------------

    def test_reference_exactly_zero(self):
        """When reference is 0, rel_error denominator is clamped to atol."""
        ref = torch.tensor([0.0])
        out = torch.tensor([1e-6])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        c, exceeds = compute_error_stats(out, ref, cfg)
        # abs_error = 1e-6, tolerance = atol = 1e-5 → within tolerance
        assert not exceeds
        # rel_error = 1e-6 / clamp(0, min=1e-5) = 1e-6 / 1e-5 = 0.1
        assert c.max_relative_error == pytest.approx(0.1, rel=1e-4)

    def test_reference_zero_output_exceeds_atol(self):
        """Error at zero reference that exceeds atol."""
        ref = torch.tensor([0.0])
        out = torch.tensor([1e-3])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        c, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds
        # rel_error = 1e-3 / 1e-5 = 100.0
        assert c.max_relative_error == pytest.approx(100.0, rel=1e-4)

    def test_reference_near_zero_small_perturbation(self):
        """Reference near zero with a perturbation smaller than atol."""
        ref = torch.tensor([1e-10])
        out = torch.tensor([1e-10 + 1e-7])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        c, exceeds = compute_error_stats(out, ref, cfg)
        # abs_error ≈ 1e-7, tolerance = atol = 1e-5 → within tolerance
        assert not exceeds
        # rel_error uses clamp(|1e-10|, min=1e-5) = 1e-5 as denominator
        # rel = 1e-7 / 1e-5 = 0.01
        assert c.max_relative_error == pytest.approx(0.01, rel=1e-2)

    def test_both_zero(self):
        """Both output and reference are zero — no error."""
        ref = torch.tensor([0.0, 0.0])
        out = torch.tensor([0.0, 0.0])
        cfg = _spec(max_atol=1e-8, max_rtol=0.0)
        c, exceeds = compute_error_stats(out, ref, cfg)
        assert c.max_absolute_error == 0.0
        assert c.max_relative_error == 0.0
        assert not exceeds

    def test_mixed_near_zero_and_large(self):
        """Mix of near-zero and large reference values."""
        ref = torch.tensor([0.0, 1.0, 1e-12, 100.0])
        out = torch.tensor([1e-6, 1.0 + 1e-6, 1e-12, 100.0 + 1e-6])
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert not exceeds

    # ------------------------------------------------------------------
    # Negative near-zero values
    # ------------------------------------------------------------------

    def test_negative_near_zero_reference(self):
        """Negative near-zero reference — clamp uses abs, so same behavior."""
        ref = torch.tensor([-1e-12])
        out = torch.tensor([1e-12])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        c, exceeds = compute_error_stats(out, ref, cfg)
        # abs_error = 2e-12, within atol=1e-5
        assert not exceeds
        # rel_error = 2e-12 / clamp(1e-12, min=1e-5) = 2e-12 / 1e-5 = 2e-7
        assert c.max_relative_error == pytest.approx(2e-7, rel=1e-2)

    # ------------------------------------------------------------------
    # Tolerance formula: |a - b| <= atol + rtol * |b|
    # ------------------------------------------------------------------

    def test_rtol_dominates_for_large_reference(self):
        """For large references, rtol term dominates tolerance."""
        ref = torch.tensor([1000.0])
        out = torch.tensor([1000.5])
        cfg = _spec(max_atol=1e-5, max_rtol=1e-3)
        # tolerance = 1e-5 + 1e-3 * 1000 = 1.00001
        # abs_error = 0.5 < 1.00001 → pass
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert not exceeds

    def test_atol_dominates_for_small_reference(self):
        """For near-zero references, atol term dominates tolerance."""
        ref = torch.tensor([1e-10])
        out = torch.tensor([1e-10 + 5e-6])
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5)
        # tolerance = 1e-5 + 1e-5 * 1e-10 ≈ 1e-5
        # abs_error = 5e-6 < 1e-5 → pass
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert not exceeds

    def test_exact_tolerance_boundary(self):
        """Error exactly at the tolerance boundary."""
        ref = torch.tensor([1.0])
        atol, rtol = 1e-3, 1e-3
        tol = atol + rtol * 1.0  # 2e-3
        # Just within
        out_in = torch.tensor([1.0 + tol - 1e-7])
        _, exceeds_in = compute_error_stats(
            out_in, ref, _spec(max_atol=atol, max_rtol=rtol)
        )
        assert not exceeds_in
        # Just outside
        out_out = torch.tensor([1.0 + tol + 1e-7])
        _, exceeds_out = compute_error_stats(
            out_out, ref, _spec(max_atol=atol, max_rtol=rtol)
        )
        assert exceeds_out

    # ------------------------------------------------------------------
    # required_matched_ratio
    # ------------------------------------------------------------------

    def test_required_matched_ratio_default_is_one(self):
        """When required_matched_ratio is 1.0, all must match."""
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = ref.clone()
        out[0] += 1.0  # one element way off
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds  # 1/3 elements fail → ratio < 1.0

    def test_partial_match_ratio_passes(self):
        """Passes when enough elements match, even if some don't."""
        ref = torch.ones(100)
        out = ref.clone()
        out[:5] += 1.0  # 5% mismatch
        cfg = _spec(required_matched_ratio=0.9)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert not exceeds

    def test_partial_match_ratio_fails(self):
        """Fails when too many elements don't match."""
        ref = torch.ones(100)
        out = ref.clone()
        out[:20] += 1.0  # 20% mismatch
        cfg = _spec(required_matched_ratio=0.9)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds

    def test_zero_matched_ratio_always_passes(self):
        """With required_matched_ratio=0, even all-wrong passes."""
        ref = torch.ones(10)
        out = ref + 100.0
        cfg = _spec(required_matched_ratio=0.0)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert not exceeds

    # ------------------------------------------------------------------
    # Subnormal / denormalized floats
    # ------------------------------------------------------------------

    def test_subnormal_reference(self):
        """Subnormal float reference values don't cause issues."""
        smallest_normal = torch.finfo(torch.float32).tiny  # ~1.17e-38
        ref = torch.tensor([smallest_normal / 2])  # subnormal
        out = torch.tensor([smallest_normal / 2 + 1e-40])
        cfg = _spec(max_atol=1e-5, max_rtol=0.0)
        c, exceeds = compute_error_stats(out, ref, cfg)
        assert not exceeds

    # ------------------------------------------------------------------
    # Scalar (0-d) tensors
    # ------------------------------------------------------------------

    def test_scalar_tensors(self):
        """0-d tensors work correctly."""
        ref = torch.tensor(5.0)
        out = torch.tensor(5.001)
        c, exceeds = compute_error_stats(out, ref, _spec(max_atol=0.01, max_rtol=0.01))
        assert not exceeds
        assert c.max_absolute_error == pytest.approx(0.001, abs=1e-4)

    # ------------------------------------------------------------------
    # dtype casting
    # ------------------------------------------------------------------

    def test_float16_inputs(self):
        """float16 inputs are upcast to float32 internally."""
        ref = torch.tensor([1.0, 2.0], dtype=torch.float16)
        out = torch.tensor([1.0, 2.0], dtype=torch.float16)
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert not exceeds

    def test_bfloat16_inputs(self):
        """bfloat16 inputs are upcast to float32 internally."""
        ref = torch.tensor([1.0, 0.0, -1.0], dtype=torch.bfloat16)
        out = ref.clone()
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert c.max_absolute_error == 0.0
        assert not exceeds

    # ------------------------------------------------------------------
    # Large tensors with sparse errors near zero
    # ------------------------------------------------------------------

    def test_large_tensor_single_near_zero_outlier(self):
        """One near-zero element with error, rest are fine."""
        ref = torch.ones(10000)
        ref[5000] = 0.0
        out = ref.clone()
        out[5000] = 1e-4  # error at the zero element
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5)
        _, exceeds = compute_error_stats(out, ref, cfg)
        # With default required_matched_ratio=1.0, this fails
        assert exceeds

    def test_large_tensor_near_zero_outlier_with_relaxed_ratio(self):
        """Same as above but with relaxed ratio — passes."""
        ref = torch.ones(10000)
        ref[5000] = 0.0
        out = ref.clone()
        out[5000] = 1e-4
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5, required_matched_ratio=0.999)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert not exceeds

    # ------------------------------------------------------------------
    # Inf / NaN behavior
    # ------------------------------------------------------------------

    def test_inf_in_output(self):
        """Inf in output produces inf abs_error and fails."""
        ref = torch.tensor([1.0])
        out = torch.tensor([float("inf")])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_nan_in_output(self):
        """NaN in output produces non-finite abs_error and fails."""
        ref = torch.tensor([1.0])
        out = torch.tensor([float("nan")])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_nan_among_valid_elements(self):
        """A single NaN in an otherwise correct tensor fails."""
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = torch.tensor([1.0, float("nan"), 3.0])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_nan_fails_even_with_relaxed_ratio(self):
        """NaN triggers early sanity check, bypassing matched-ratio tolerance."""
        ref = torch.ones(100)
        out = ref.clone()
        out[0] = float("nan")
        cfg = _spec(required_matched_ratio=0.0)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds

    def test_inf_among_valid_elements(self):
        """A single spurious inf in an otherwise correct tensor fails."""
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = torch.tensor([1.0, float("inf"), 3.0])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_negative_inf_in_output(self):
        """Negative inf in output fails when reference is finite."""
        ref = torch.tensor([1.0])
        out = torch.tensor([float("-inf")])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_matching_inf_fails(self):
        """Matching infinities (both +inf) are still incorrect."""
        ref = torch.tensor([1.0, float("inf"), 3.0])
        out = torch.tensor([1.0, float("inf"), 3.0])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_matching_neg_inf_fails(self):
        """Matching negative infinities are still incorrect."""
        ref = torch.tensor([float("-inf"), 2.0])
        out = torch.tensor([float("-inf"), 2.0])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_inf_in_reference_only(self):
        """Inf in reference (but not output) is also incorrect."""
        ref = torch.tensor([float("inf"), 2.0])
        out = torch.tensor([1.0, 2.0])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_nan_in_reference_only(self):
        """NaN in reference (but not output) is also incorrect."""
        ref = torch.tensor([1.0, float("nan")])
        out = torch.tensor([1.0, 2.0])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_mismatched_inf_sign_fails(self):
        """+inf vs -inf at the same position fails."""
        ref = torch.tensor([float("inf")])
        out = torch.tensor([float("-inf")])
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_all_nan_output(self):
        """Entirely NaN output fails."""
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = torch.full((3,), float("nan"))
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    def test_all_inf_output(self):
        """Entirely inf output fails when reference is finite."""
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = torch.full((3,), float("inf"))
        _, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds

    # ------------------------------------------------------------------
    # allow_negative_inf
    # ------------------------------------------------------------------

    def test_matching_neg_inf_passes_when_allowed(self):
        """Both tensors have -inf at the same position — accepted."""
        ref = torch.tensor([float("-inf"), 2.0])
        out = torch.tensor([float("-inf"), 2.0])
        cfg = _spec(allow_negative_inf=True)
        c, exceeds = compute_error_stats(out, ref, cfg)
        assert c.max_absolute_error == 0.0
        assert c.max_relative_error == 0.0
        assert not exceeds

    def test_matching_neg_inf_still_fails_when_not_allowed(self):
        """Default behavior: matching -inf fails."""
        ref = torch.tensor([float("-inf"), 2.0])
        out = torch.tensor([float("-inf"), 2.0])
        cfg = _spec(allow_negative_inf=False)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds

    def test_all_matching_neg_inf_passes_when_allowed(self):
        """All elements are matching -inf — treated as fully correct."""
        ref = torch.full((5,), float("-inf"))
        out = torch.full((5,), float("-inf"))
        cfg = _spec(allow_negative_inf=True)
        c, exceeds = compute_error_stats(out, ref, cfg)
        assert c.max_absolute_error == 0.0
        assert c.max_relative_error == 0.0
        assert not exceeds

    def test_neg_inf_only_in_output_fails_when_allowed(self):
        """-inf in output but not reference still fails."""
        ref = torch.tensor([1.0, 2.0])
        out = torch.tensor([float("-inf"), 2.0])
        cfg = _spec(allow_negative_inf=True)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds

    def test_neg_inf_only_in_reference_fails_when_allowed(self):
        """-inf in reference but not output still fails."""
        ref = torch.tensor([float("-inf"), 2.0])
        out = torch.tensor([1.0, 2.0])
        cfg = _spec(allow_negative_inf=True)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds

    def test_pos_inf_still_fails_when_neg_inf_allowed(self):
        """+inf is not affected by allow_negative_inf."""
        ref = torch.tensor([float("inf"), 2.0])
        out = torch.tensor([float("inf"), 2.0])
        cfg = _spec(allow_negative_inf=True)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds

    def test_nan_still_fails_when_neg_inf_allowed(self):
        """NaN is not affected by allow_negative_inf."""
        ref = torch.tensor([1.0, 2.0])
        out = torch.tensor([float("nan"), 2.0])
        cfg = _spec(allow_negative_inf=True)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds

    def test_neg_inf_allowed_with_finite_errors(self):
        """Matching -inf excluded; error computed on remaining elements."""
        ref = torch.tensor([float("-inf"), 1.0, 2.0])
        out = torch.tensor([float("-inf"), 1.0 + 1e-6, 2.0 + 1e-6])
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5, allow_negative_inf=True)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert not exceeds

    def test_neg_inf_allowed_mixed_inf_sign_fails(self):
        """-inf vs +inf at same position fails even when allow_negative_inf."""
        ref = torch.tensor([float("-inf")])
        out = torch.tensor([float("inf")])
        cfg = _spec(allow_negative_inf=True)
        _, exceeds = compute_error_stats(out, ref, cfg)
        assert exceeds

    # ------------------------------------------------------------------
    # max_error_cap (cuDNN pattern)
    # ------------------------------------------------------------------

    def test_max_error_cap_no_cap(self):
        """Without cap, large outlier errors pass if matched ratio is met."""
        ref = torch.ones(100)
        out = ref.clone()
        out[0] = 100.0  # huge error on one element
        cfg = _spec(max_atol=1e-5, max_rtol=1e-5, required_matched_ratio=0.95)
        _, exceeds = compute_error_stats(out, ref, cfg)
        # 99/100 match → ratio=0.99 ≥ 0.95 → passes (no cap)
        assert not exceeds

    def test_max_error_cap_passes(self):
        """Error below cap passes normally."""
        ref = torch.ones(100)
        out = ref.clone()
        out[0] = 1.5  # abs_error = 0.5, below cap of 1.0
        cfg = _spec(
            max_atol=1e-5,
            max_rtol=1e-5,
            required_matched_ratio=0.95,
            max_error_cap=1.0,
        )
        _, exceeds = compute_error_stats(out, ref, cfg)
        # 99/100 match → ratio=0.99 ≥ 0.95, max_abs=0.5 ≤ 1.0 → passes
        assert not exceeds

    def test_max_error_cap_fails(self):
        """Error above cap fails even with good matched ratio."""
        ref = torch.ones(100)
        out = ref.clone()
        out[0] = 100.0  # abs_error = 99.0, way above cap of 1.0
        cfg = _spec(
            max_atol=1e-5,
            max_rtol=1e-5,
            required_matched_ratio=0.95,
            max_error_cap=1.0,
        )
        _, exceeds = compute_error_stats(out, ref, cfg)
        # 99/100 match → ratio=0.99 ≥ 0.95 BUT max_abs=99.0 > 1.0 → fails
        assert exceeds

    def test_max_error_cap_boundary(self):
        """Cap uses strict greater-than (not strictly greater)."""
        ref = torch.tensor([1.0])
        out = torch.tensor([2.0])  # abs_error = 1.0, exactly at cap
        cfg = _spec(
            max_atol=1e-5,
            max_rtol=1e-5,
            required_matched_ratio=0.0,
            max_error_cap=1.0,
        )
        c, exceeds = compute_error_stats(out, ref, cfg)
        assert c.max_absolute_error == pytest.approx(1.0, abs=1e-6)
        # 1.0 is NOT > 1.0, so cap should not trigger
        assert not exceeds

    # ------------------------------------------------------------------
    # has_nan / has_inf boolean flags (JSON compliance)
    # ------------------------------------------------------------------

    def test_nan_output_sets_has_nan_flag(self):
        """NaN in output returns Correctness with has_nan=True and finite errors."""
        ref = torch.tensor([1.0, 2.0])
        out = torch.tensor([1.0, float("nan")])
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds
        assert c.has_nan is True
        assert c.has_inf is False
        assert c.max_absolute_error == 0.0
        assert c.max_relative_error == 0.0

    def test_inf_output_sets_has_inf_flag(self):
        """Inf in output returns Correctness with has_inf=True and finite errors."""
        ref = torch.tensor([1.0])
        out = torch.tensor([float("inf")])
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds
        assert c.has_inf is True
        assert c.has_nan is False
        assert c.max_absolute_error == 0.0
        assert c.max_relative_error == 0.0

    def test_nan_in_reference_sets_has_nan_flag(self):
        """NaN in reference also triggers has_nan."""
        ref = torch.tensor([float("nan"), 2.0])
        out = torch.tensor([1.0, 2.0])
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds
        assert c.has_nan is True

    def test_mixed_nan_inf_prefers_has_nan(self):
        """When both NaN and Inf are present, has_nan takes priority."""
        ref = torch.tensor([float("inf"), 2.0])
        out = torch.tensor([float("nan"), 2.0])
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds
        assert c.has_nan is True
        assert c.has_inf is False

    def test_all_zeros_output_returns_finite_errors(self):
        """All-zeros output (nonzero ref) returns finite error values, no flags."""
        ref = torch.tensor([1.0, 2.0, 3.0])
        out = torch.zeros(3)
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds
        assert c.has_nan is False
        assert c.has_inf is False
        assert c.max_absolute_error > 0  # ref_norm
        assert c.max_relative_error > 0

    def test_correctness_error_values_are_json_safe(self):
        """All Correctness instances from compute_error_stats have JSON-safe floats."""
        import json
        import math

        cases = [
            (torch.tensor([float("nan")]), torch.tensor([1.0])),
            (torch.tensor([float("inf")]), torch.tensor([1.0])),
            (torch.tensor([float("-inf")]), torch.tensor([1.0])),
            (torch.zeros(3), torch.ones(3)),
        ]
        for out, ref in cases:
            c, _ = compute_error_stats(out, ref, _spec())
            assert not math.isnan(c.max_absolute_error), (
                f"NaN in max_absolute_error for {out}"
            )
            assert not math.isnan(c.max_relative_error), (
                f"NaN in max_relative_error for {out}"
            )
            assert not math.isinf(c.max_absolute_error), (
                f"Inf in max_absolute_error for {out}"
            )
            assert not math.isinf(c.max_relative_error), (
                f"Inf in max_relative_error for {out}"
            )
            # Must serialize without allow_nan
            data = c.model_dump(mode="json")
            json.dumps(data, allow_nan=False)  # must not raise

    def test_compute_error_stats_nan_returns_finite(self):
        """compute_error_stats with NaN output returns finite abs/rel error values."""
        import math

        ref = torch.tensor([1.0, 2.0])
        out = torch.tensor([1.0, float("nan")])
        c, exceeds = compute_error_stats(out, ref, _spec())
        assert exceeds
        assert not math.isnan(c.max_absolute_error), "max_abs should be finite"
        assert not math.isnan(c.max_relative_error), "max_rel should be finite"
        assert not math.isinf(c.max_absolute_error), "max_abs should be finite"
        assert not math.isinf(c.max_relative_error), "max_rel should be finite"


class TestSetSeed:
    """Tests that set_seed produces reproducible GPU tensor generation."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def test_set_seed_reproducible_rand(self):
        """torch.rand on GPU gives identical results after set_seed with same seed."""
        set_seed(200)
        a = torch.rand(1024, 1024, device=self.DEVICE)
        set_seed(200)
        b = torch.rand(1024, 1024, device=self.DEVICE)
        assert torch.equal(a, b)

    def test_set_seed_reproducible_randn(self):
        """torch.randn on GPU gives identical results after set_seed with same seed."""
        set_seed(200)
        a = torch.randn(512, 512, device=self.DEVICE)
        set_seed(200)
        b = torch.randn(512, 512, device=self.DEVICE)
        assert torch.equal(a, b)

    def test_set_seed_different_seeds_differ(self):
        """Different seeds produce different tensors."""
        set_seed(200)
        a = torch.rand(256, 256, device=self.DEVICE)
        set_seed(999)
        b = torch.rand(256, 256, device=self.DEVICE)
        assert not torch.equal(a, b)

    def test_set_seed_from_benchmark_config(self):
        """BenchmarkConfig.seed defaults to 200 and produces reproducible results."""
        from sol_execbench.core.bench.config import BenchmarkConfig

        cfg = BenchmarkConfig()
        set_seed(cfg.seed)
        a = torch.rand(128, 128, device=self.DEVICE)
        set_seed(cfg.seed)
        b = torch.rand(128, 128, device=self.DEVICE)
        assert torch.equal(a, b)
        assert cfg.seed == 200
