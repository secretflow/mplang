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

echo "================================"
echo "Running Device-level Tutorials"
echo "================================"

echo -e "\n[Device 00] Device Basics..."
uv run python tutorials/00_device_basics.py

echo -e "\n[Device 01] Function Decorator..."
uv run python tutorials/01_function_decorator.py

echo -e "\n[Device 02] Simulation and Driver..."
uv run python tutorials/02_simulation_and_driver.py sim

echo -e "\n[Device 03] Run JAX..."
uv run python tutorials/03_run_jax.py

echo -e "\n[Device 04] IR Dump and Analysis..."
uv run python tutorials/04_ir_dump_and_analysis.py

echo -e "\n[Device 05] Run SQL..."
uv run python tutorials/05_run_sql.py

echo -e "\n[Device 06] Hybrid JAX/SQL Pipeline..."
uv run python tutorials/06_pipeline.py

echo -e "\n[Device 07] Stax Neural Network..."
uv run python tutorials/07_stax_nn.py

echo -e "\n[Device 08] Logging..."
uv run python tutorials/08_logging.py

echo -e "\n================================"
echo "All tutorials completed successfully!"
echo "================================"
