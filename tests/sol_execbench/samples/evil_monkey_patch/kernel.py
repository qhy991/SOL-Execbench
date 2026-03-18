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

"""Evil kernel: patches torch.cuda.Event.elapsed_time at module import time.

The monkey-patch occurs before check_monkey_patch() runs, but after
reward_hack.py captures the original function identity at its own
module load time.  The driver will detect the discrepancy and emit
a REWARD_HACK trace.
"""

import torch

# Patch at module level — before the driver calls check_monkey_patch().
torch.cuda.Event.elapsed_time = lambda self, end: 0.0


def run(x, y):
    return x + y
