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

"""
Test PRNG key handling in MPLang functions with fixed approach.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import mplang as mp


class TestMPLangPRNGFixed:
    """Test suite for PRNG key handling in MPLang functions with fixed approach."""

    def test_mplang_prng_key_split_fixed(self):
        """Test MPLang function with PRNG key splitting using fixed approach."""
        
        @mp.function
        def key_split_function():
            # Create a PRNG key at party 0
            key = mp.run_jax_at(0, jax.random.PRNGKey, 42)
            
            # Split the key using the workaround approach
            def split_kernel(k):
                return jax.random.split(k)
            
            keys = mp.run_jax_at(0, split_kernel, key)
            key1 = mp.run_jax_at(0, lambda ks: ks[0], keys)
            key2 = mp.run_jax_at(0, lambda ks: ks[1], keys)
            
            # Use the keys to generate random numbers
            rand1 = mp.run_jax_at(0, jax.random.normal, key1, (3,))
            rand2 = mp.run_jax_at(0, jax.random.normal, key2, (3,))
            
            return rand1, rand2

        # Create a simulator
        sim = mp.Simulator.simple(1)
        
        # Compile the function
        compiled = mp.compile(sim, key_split_function)
        
        # Verify that compilation was successful
        assert compiled is not None
        
    def test_mplang_prng_with_dropout_like_operation_fixed(self):
        """Test MPLang function with dropout-like operation using PRNG key with fixed approach."""
        
        @mp.function
        def dropout_function():
            # Create a PRNG key at party 0
            key = mp.run_jax_at(0, jax.random.PRNGKey, 42)
            
            # Create input data
            x = mp.constant(jnp.ones((4,)))
            
            # Dropout-like operation
            def dropout_op(key, x, rate=0.5):
                """A simplified dropout-like function that uses a PRNG key."""
                # Generate random numbers using the key
                rand = jax.random.uniform(key, x.shape)
                # Create a mask
                mask = rand > rate
                # Apply the mask
                return x * mask / (1.0 - rate)
            
            result = mp.run_jax_at(0, dropout_op, key, x)
            
            return result

        # Create a simulator
        sim = mp.Simulator.simple(1)
        
        # Compile the function
        compiled = mp.compile(sim, dropout_function)
        
        # Verify that compilation was successful
        assert compiled is not None
        
    def test_mplang_prng_key_chaining_fixed(self):
        """Test MPLang function with chained PRNG key operations using fixed approach."""
        
        @mp.function
        def key_chain_function():
            # Create a PRNG key at party 0
            key = mp.run_jax_at(0, jax.random.PRNGKey, 42)
            
            # Chain multiple key operations using the workaround approach
            def split_kernel(k):
                return jax.random.split(k)
            
            keys1 = mp.run_jax_at(0, split_kernel, key)
            key1 = mp.run_jax_at(0, lambda ks: ks[0], keys1)
            key = mp.run_jax_at(0, lambda ks: ks[1], keys1)
            
            keys2 = mp.run_jax_at(0, split_kernel, key)
            key2 = mp.run_jax_at(0, lambda ks: ks[0], keys2)
            key = mp.run_jax_at(0, lambda ks: ks[1], keys2)
            
            keys3 = mp.run_jax_at(0, split_kernel, key)
            key3 = mp.run_jax_at(0, lambda ks: ks[0], keys3)
            key4 = mp.run_jax_at(0, lambda ks: ks[1], keys3)
            
            # Use the keys to generate different random numbers
            rand1 = mp.run_jax_at(0, jax.random.normal, key1, (2,))
            rand2 = mp.run_jax_at(0, jax.random.normal, key2, (2,))
            rand3 = mp.run_jax_at(0, jax.random.uniform, key3, (2,))
            rand4 = mp.run_jax_at(0, jax.random.uniform, key4, (2,))
            
            return rand1, rand2, rand3, rand4

        # Create a simulator
        sim = mp.Simulator.simple(1)
        
        # Compile the function
        compiled = mp.compile(sim, key_chain_function)
        
        # Verify that compilation was successful
        assert compiled is not None


if __name__ == "__main__":
    pytest.main([__file__])
