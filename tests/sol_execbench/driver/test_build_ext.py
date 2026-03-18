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


"""Tests for the build_ext.py template.

The template reads ``solution.json`` from the working directory and extracts
compile options via the Solution model.  Tests here exercise the template by
exec-ing it in a controlled tmp directory with ``torch.utils.cpp_extension``
and ``sol_execbench.core`` mocked out.
"""

import ast
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "sol_execbench"
    / "driver"
    / "templates"
    / "build_ext.py"
)


def _make_solution_json(compile_options: dict | None = None) -> str:
    """Create a minimal valid solution JSON string with given compile_options."""
    solution = {
        "name": "test_solution",
        "definition": "test_def",
        "author": "test",
        "spec": {
            "languages": ["cuda_cpp"],
            "target_hardware": ["LOCAL"],
            "entry_point": "main.cpp::run",
            "destination_passing_style": True,
            "compile_options": compile_options,
        },
        "sources": [{"path": "main.cpp", "content": "// test source"}],
    }
    return json.dumps(solution)


def _exec_build_ext(
    cwd: Path,
    compile_options: dict | None = None,
    *,
    cutlass_dir: str | None = None,
) -> MagicMock:
    """Write solution.json and execute the build_ext template in *cwd* with ext.load mocked.

    Returns the ``ext.load`` mock so callers can inspect how it was called.
    """
    # Write solution.json
    (cwd / "solution.json").write_text(_make_solution_json(compile_options))

    script = _TEMPLATE_PATH.read_text()

    # Stub torch.utils.cpp_extension so we don't need a real torch install
    mock_ext = MagicMock()
    fake_torch = types.ModuleType("torch")
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_cpp = types.ModuleType("torch.utils.cpp_extension")
    fake_torch_utils_cpp.load = mock_ext.load
    fake_torch.utils = fake_torch_utils
    fake_torch_utils.cpp_extension = fake_torch_utils_cpp

    saved_modules = {}
    for mod_name in ("torch", "torch.utils", "torch.utils.cpp_extension"):
        saved_modules[mod_name] = sys.modules.get(mod_name)

    sys.modules["torch"] = fake_torch
    sys.modules["torch.utils"] = fake_torch_utils
    sys.modules["torch.utils.cpp_extension"] = fake_torch_utils_cpp

    old_cwd = os.getcwd()
    env_backup = os.environ.get("CUTLASS_DIR")
    try:
        os.chdir(cwd)
        if cutlass_dir is not None:
            os.environ["CUTLASS_DIR"] = cutlass_dir
        elif "CUTLASS_DIR" in os.environ:
            del os.environ["CUTLASS_DIR"]
        exec(compile(script, "build_ext.py", "exec"), {"__builtins__": __builtins__})
    finally:
        os.chdir(old_cwd)
        if env_backup is not None:
            os.environ["CUTLASS_DIR"] = env_backup
        elif "CUTLASS_DIR" in os.environ:
            del os.environ["CUTLASS_DIR"]
        for mod_name, orig in saved_modules.items():
            if orig is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = orig

    return mock_ext


class TestTemplateAST:
    """Verify the raw template is valid Python."""

    def test_raw_template_parses(self):
        source = _TEMPLATE_PATH.read_text()
        tree = ast.parse(source, filename="build_ext.py")
        assert isinstance(tree, ast.Module)
        assert len(tree.body) > 0

    def test_template_has_expected_imports(self):
        source = _TEMPLATE_PATH.read_text()
        tree = ast.parse(source, filename="build_ext.py")
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                import_names.append(node.module)
        assert "os" in import_names
        assert "torch.utils.cpp_extension" in import_names
        assert "sol_execbench.core" in import_names


class TestSourceDiscovery:
    """Tests for source file collection logic."""

    @pytest.mark.parametrize("suffix", [".cu", ".cpp", ".cc", ".cxx", ".c"])
    def test_collects_supported_extensions(self, tmp_path, suffix):
        (tmp_path / f"kernel{suffix}").write_text("// source")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        sources = mock.load.call_args.kwargs["sources"]
        assert len(sources) == 1
        assert sources[0].endswith(suffix)

    def test_ignores_non_source_files(self, tmp_path):
        (tmp_path / "kernel.cu").write_text("// source")
        (tmp_path / "readme.md").write_text("docs")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "helper.py").write_text("pass")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        sources = mock.load.call_args.kwargs["sources"]
        assert len(sources) == 1

    def test_collects_multiple_sources(self, tmp_path):
        (tmp_path / "kernel.cu").write_text("// cuda")
        (tmp_path / "utils.cpp").write_text("// cpp")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        sources = mock.load.call_args.kwargs["sources"]
        assert len(sources) == 2

    def test_no_sources_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="No CUDA/C\\+\\+ source files"):
            _exec_build_ext(tmp_path)


class TestCompileOptions:
    """Tests for compile option extraction."""

    def test_default_cuda_cflags(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        cuda_cflags = mock.load.call_args.kwargs["extra_cuda_cflags"]
        assert cuda_cflags == ["-O3", "--use_fast_math"]

    def test_custom_cuda_cflags_appended(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, {"cuda_cflags": ["-arch=sm_90a"]})
        cuda_cflags = mock.load.call_args.kwargs["extra_cuda_cflags"]
        assert cuda_cflags == ["-O3", "--use_fast_math", "-arch=sm_90a"]

    def test_cflags_default_empty(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        assert mock.load.call_args.kwargs["extra_cflags"] == []

    def test_custom_cflags(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, {"cflags": ["-Wall"]})
        assert mock.load.call_args.kwargs["extra_cflags"] == ["-Wall"]

    def test_custom_ld_flags(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, {"ld_flags": ["-lcublas"]})
        assert mock.load.call_args.kwargs["extra_ldflags"] == ["-lcublas"]


class TestIncludePaths:
    """Tests for extra_include_paths construction."""

    def test_default_cutlass_dir(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        includes = mock.load.call_args.kwargs["extra_include_paths"]
        assert str(tmp_path) in includes
        assert "/usr/local/cutlass/include" in includes
        assert "/usr/local/cutlass/tools/util/include" in includes

    def test_custom_cutlass_dir(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path, cutlass_dir="/opt/cutlass")
        includes = mock.load.call_args.kwargs["extra_include_paths"]
        assert "/opt/cutlass/include" in includes
        assert "/opt/cutlass/tools/util/include" in includes


class TestSoRename:
    """Tests for the .so rename logic after compilation."""

    def test_renames_suffixed_so(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        suffixed = tmp_path / "benchmark_kernel.cpython-312-x86_64-linux-gnu.so"
        suffixed.write_bytes(b"ELF-fake")
        _exec_build_ext(tmp_path)
        assert (tmp_path / "benchmark_kernel.so").exists()
        assert not suffixed.exists()

    def test_already_named_correctly(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"ELF-fake")
        _exec_build_ext(tmp_path)
        assert (tmp_path / "benchmark_kernel.so").exists()

    def test_no_so_raises(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        with pytest.raises(FileNotFoundError, match="benchmark_kernel.so not produced"):
            _exec_build_ext(tmp_path)


class TestExtLoad:
    """Tests for ext.load call arguments."""

    def test_load_called_with_correct_name(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        assert mock.load.call_args.kwargs["name"] == "benchmark_kernel"

    def test_build_directory_is_cwd(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        assert mock.load.call_args.kwargs["build_directory"] == str(tmp_path)

    def test_verbose_enabled(self, tmp_path):
        (tmp_path / "k.cu").write_text("")
        (tmp_path / "benchmark_kernel.so").write_bytes(b"fake")
        mock = _exec_build_ext(tmp_path)
        assert mock.load.call_args.kwargs["verbose"] is True
