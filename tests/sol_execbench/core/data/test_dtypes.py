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


"""Tests for sol_execbench.core.data.dtypes and the re-export via utils."""

import torch
import pytest

from sol_execbench.core.data.dtypes import (
    dtype_str_to_python_dtype,
    dtype_str_to_torch_dtype,
    is_dtype_integer,
)


class TestDtypeStrToTorchDtype:
    """Tests for dtype_str_to_torch_dtype (canonical source in data/dtypes.py)."""

    @pytest.mark.parametrize(
        "dtype_str, expected",
        [
            ("float64", torch.float64),
            ("float32", torch.float32),
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
            ("float8_e4m3fn", torch.float8_e4m3fn),
            ("float8_e5m2", torch.float8_e5m2),
            ("float4_e2m1", torch.float4_e2m1fn_x2),
            ("float4_e2m1fn_x2", torch.float4_e2m1fn_x2),
            ("int64", torch.int64),
            ("int32", torch.int32),
            ("int16", torch.int16),
            ("int8", torch.int8),
            ("bool", torch.bool),
        ],
    )
    def test_valid_dtype(self, dtype_str, expected):
        assert dtype_str_to_torch_dtype(dtype_str) is expected

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_str_to_torch_dtype("complex128")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="None or empty"):
            dtype_str_to_torch_dtype("")


class TestDtypeStrToPythonDtype:
    """Tests for dtype_str_to_python_dtype."""

    @pytest.mark.parametrize(
        "dtype_str, expected",
        [
            ("float32", float),
            ("float16", float),
            ("bfloat16", float),
            ("float8_e4m3fn", float),
            ("float8_e5m2", float),
            ("float4_e2m1", float),
            ("float4_e2m1fn_x2", float),
            ("int64", int),
            ("int32", int),
            ("int16", int),
            ("int8", int),
            ("bool", bool),
        ],
    )
    def test_valid_dtype(self, dtype_str, expected):
        assert dtype_str_to_python_dtype(dtype_str) is expected

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_str_to_python_dtype("complex128")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="None or empty"):
            dtype_str_to_python_dtype("")


class TestIsDtypeInteger:
    """Tests for is_dtype_integer."""

    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool],
    )
    def test_integer_dtypes(self, dtype):
        assert is_dtype_integer(dtype) is True

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16],
    )
    def test_float_dtypes(self, dtype):
        assert is_dtype_integer(dtype) is False

