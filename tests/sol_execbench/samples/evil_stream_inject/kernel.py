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

"""Evil kernel: enqueues heavy work on a non-default CUDA stream.

CUDA event timing (used by time_runnable) records only the default stream,
so the fast x + y appears blazing fast.  check_stream_injection() re-runs
the kernel and calls torch.cuda.synchronize(), which must wait for the side
stream's 50 matmuls — exposing a wall-clock time far exceeding the measured
latency multiplied by the threshold factor.
"""

import torch


def run(x, y):
    result = x + y
    if x.is_cuda:
        _side = torch.cuda.Stream()
        with torch.cuda.stream(_side):
            _a = torch.randn(4096, 4096, device=x.device, dtype=torch.float32)
            _b = torch.randn(4096, 4096, device=x.device, dtype=torch.float32)
            for _ in range(50):
                torch.mm(_a, _b)
    return result
