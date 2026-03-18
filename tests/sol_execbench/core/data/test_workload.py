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


"""Tests for sol_execbench.core.data.workload.Workload."""

from sol_execbench.core.data.workload import Workload


def _wkl(**inputs):
    return Workload(uuid="test-uuid", axes={}, inputs=inputs)


class TestGetScalarInputs:
    def test_no_scalars_returns_empty(self):
        wkl = _wkl(a={"type": "random"})
        assert wkl.get_scalar_inputs() == {}

    def test_single_int_scalar(self):
        wkl = _wkl(x={"type": "scalar", "value": 42})
        assert wkl.get_scalar_inputs() == {"x": 42}

    def test_single_float_scalar(self):
        wkl = _wkl(scale={"type": "scalar", "value": 0.5})
        assert wkl.get_scalar_inputs() == {"scale": 0.5}

    def test_multiple_scalars(self):
        wkl = _wkl(
            a={"type": "scalar", "value": 1},
            b={"type": "scalar", "value": 2},
        )
        assert wkl.get_scalar_inputs() == {"a": 1, "b": 2}

    def test_mixed_inputs_returns_only_scalars(self):
        wkl = _wkl(
            a={"type": "random"},
            b={"type": "scalar", "value": 7},
            c={"type": "safetensors", "path": "f.safetensors", "tensor_key": "k"},
        )
        assert wkl.get_scalar_inputs() == {"b": 7}

    def test_all_custom_returns_empty(self):
        wkl = _wkl(a={"type": "custom"}, b={"type": "custom"})
        assert wkl.get_scalar_inputs() == {}
