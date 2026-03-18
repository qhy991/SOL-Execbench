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

"""Evil kernel: returns a torch.Tensor subclass instead of a plain torch.Tensor.

check_lazy_outputs() uses strict type() equality (not isinstance), so any
subclass — including this _FakeTensor — is rejected with REWARD_HACK.
"""

import torch


class _FakeTensor(torch.Tensor):
    pass


def run(x, y):
    return (x + y).as_subclass(_FakeTensor)
