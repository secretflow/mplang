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
uv run python tutorials/v1/device/00_device_basics.py

echo -e "\n[Device 01] Function Decorator..."
uv run python tutorials/v1/device/01_function_decorator.py

echo -e "\n[Device 02] Simulation and Driver..."
uv run python tutorials/v1/device/02_simulation_and_driver.py sim

echo -e "\n[Device 03] Run JAX..."
uv run python tutorials/v1/device/03_run_jax.py

echo -e "\n[Device 04] Run SQL..."
uv run python tutorials/v1/device/04_run_sql.py

echo -e "\n[Device 05] Hybrid JAX/SQL IO..."
uv run python tutorials/v1/device/05_pipeline.py

echo -e "\n[Device 06] IR Dump and Analysis..."
uv run python tutorials/v1/device/06_ir_dump_and_analysis.py

echo -e "\n================================"
echo "Running Simp-level Tutorials"
echo "================================"

echo -e "\n[Simp 00] Basic..."
uv run python tutorials/v1/simp/00_basic.py

echo -e "\n[Simp 01] Condition..."
uv run python tutorials/v1/simp/01_condition.py

echo -e "\n[Simp 02] While Loop..."
uv run python tutorials/v1/simp/02_whileloop.py

echo -e "\n[Simp 03] STDIO..."
uv run python tutorials/v1/simp/03_stdio.py

echo -e "\n[Simp 04] PHE (Paillier Homomorphic Encryption)..."
uv run python tutorials/v1/simp/04_phe.py

echo -e "\n[Simp 05] TEE (Trusted Execution Environment)..."
uv run python tutorials/v1/simp/05_tee.py

echo -e "\n[Simp 06] FHE (Fully Homomorphic Encryption)..."
uv run python tutorials/v1/simp/06_fhe.py

echo -e "\n[Simp 07] Advanced..."
uv run python tutorials/v1/simp/07_advanced.py

echo -e "\n[Simp 08] Simple Secret Sharing..."
uv run python tutorials/v1/simp/08_simple_secret_sharing.py

echo -e "\n================================"
echo "All tutorials completed successfully!"
echo "================================"
