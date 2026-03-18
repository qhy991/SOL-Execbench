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


"""Tests for sol_execbench.core.data.definition.Definition."""

import pytest

from sol_execbench.core.data.definition import Definition

_REFERENCE = "def run(a): return a"


def _make(**overrides):
    base = dict(
        name="op",
        op_type="gemm",
        axes={"N": {"type": "var"}},
        inputs={"a": {"shape": ["N"], "dtype": "float32"}},
        outputs={"b": {"shape": ["N"], "dtype": "float32"}},
        reference=_REFERENCE,
    )
    base.update(overrides)
    return Definition(**base)


# ── get_resolved_axes_values ──────────────────────────────────────────────────


class TestGetResolvedAxesValues:
    def test_var_axis_passed_through(self):
        d = _make()
        result = d.get_resolved_axes_values({"N": 8})
        assert result["N"] == 8

    def test_const_axis_included(self):
        d = _make(
            axes={"N": {"type": "var"}, "C": {"type": "const", "value": 16}},
            inputs={"a": {"shape": ["N", "C"], "dtype": "float32"}},
            outputs={"b": {"shape": ["N"], "dtype": "float32"}},
        )
        result = d.get_resolved_axes_values({"N": 4})
        assert result["N"] == 4
        assert result["C"] == 16

    def test_expr_axis_evaluated(self):
        d = _make(
            axes={"N": {"type": "var"}, "N2": {"type": "expr", "expression": "N * 2"}},
            outputs={"b": {"shape": ["N2"], "dtype": "float32"}},
        )
        result = d.get_resolved_axes_values({"N": 5})
        assert result["N2"] == 10

    def test_all_axis_types_combined(self):
        d = _make(
            axes={
                "N": {"type": "var"},
                "C": {"type": "const", "value": 3},
                "NC": {"type": "expr", "expression": "N * C"},
            },
            inputs={"a": {"shape": ["N", "C"], "dtype": "float32"}},
            outputs={"b": {"shape": ["NC"], "dtype": "float32"}},
        )
        result = d.get_resolved_axes_values({"N": 5})
        assert result == {"N": 5, "C": 3, "NC": 15}

    def test_var_overrides_would_be_from_workload(self):
        """Different var values produce different resolved dicts."""
        d = _make()
        assert d.get_resolved_axes_values({"N": 4})["N"] == 4
        assert d.get_resolved_axes_values({"N": 128})["N"] == 128


# ── validators ────────────────────────────────────────────────────────────────


class TestDefinitionValidators:
    def test_missing_run_function_raises(self):
        with pytest.raises(ValueError, match="run"):
            _make(reference="def helper(): pass")

    def test_invalid_python_reference_raises(self):
        with pytest.raises(ValueError):
            _make(reference="def run(: bad syntax")

    def test_undefined_axis_in_input_shape_raises(self):
        with pytest.raises(ValueError, match="undefined"):
            _make(inputs={"a": {"shape": ["UNDEFINED_AXIS"], "dtype": "float32"}})

    def test_overlapping_input_output_names_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            _make(
                inputs={"a": {"shape": ["N"], "dtype": "float32"}},
                outputs={"a": {"shape": ["N"], "dtype": "float32"}},
            )

    def test_input_name_colliding_with_axis_name_raises(self):
        with pytest.raises(ValueError, match="not allowed to be an axis"):
            _make(
                axes={"N": {"type": "var"}},
                inputs={"N": {"shape": ["N"], "dtype": "float32"}},
                reference="def run(N): return N",
            )

    def test_missing_custom_inputs_entrypoint_in_reference_raises(self):
        with pytest.raises(ValueError, match="custom_inputs_entrypoint"):
            _make(
                custom_inputs_entrypoint="nonexistent_fn",
                reference=_REFERENCE,
            )

    def test_valid_custom_inputs_entrypoint_accepted(self):
        d = _make(
            custom_inputs_entrypoint="gen",
            reference="def run(a): return a\ndef gen(axes, device): return {}",
        )
        assert d.custom_inputs_entrypoint == "gen"
