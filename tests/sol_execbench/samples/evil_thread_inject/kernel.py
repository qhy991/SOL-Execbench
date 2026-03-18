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

"""Evil kernel: spawns a long-lived daemon thread on every run() call.

The driver captures threading.active_count() before and after the user
call.  The lingering daemon thread raises the count, triggering
check_thread_injection() and a REWARD_HACK trace.
"""

import threading
import time


def run(x, y):
    threading.Thread(target=lambda: time.sleep(10), daemon=True).start()
    return x + y
