#!/bin/bash
#
# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

uv run python tutorials/0_basic.py
uv run python tutorials/1_condition.py
uv run python tutorials/2_whileloop.py
uv run python tutorials/3_device.py
uv run python tutorials/4_simulation.py sim
uv run python tutorials/5_ir_dump.py
uv run python tutorials/6_advanced.py
uv run python tutorials/7_stdio.py
uv run python tutorials/8_phe.py
uv run python tutorials/9_tee.py
uv run python tutorials/10_analysis.py
