# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common constants for MPC protocols."""

# Golden Ratio for 64-bit Multiplicative Hashing
# Closest integer to 2^64 / phi
# Golden Ratio for 64-bit Multiplicative Hashing
# Closest integer to 2^64 / phi
GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15

# LCG Constants
LCG_ADDEND = 0x14650FB0739D0383
LCG_MULTIPLIER = 0x27D4EB2F165667C5

# SplitMix64 Constants (Gamma values)
SPLITMIX64_GAMMA_1 = 0xBF58476D1CE4E5B9
SPLITMIX64_GAMMA_2 = 0x94D049BB133111EB
SPLITMIX64_GAMMA_3 = 0xFF51AFD7ED558CCD
SPLITMIX64_GAMMA_4 = 0xC4CEB9FE1A85EC53

# Arbitrary Constants (Nothing-up-my-sleeve numbers)
# Fractional part of PI (Hex)
PI_FRAC_1 = 0x243F6A8885A308D3
PI_FRAC_2 = 0x13198A2E03707344

# Fractional part of E (Hex)
E_FRAC_1 = 0xA4093822299F31D0
