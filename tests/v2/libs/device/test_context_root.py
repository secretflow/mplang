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

import unittest

import mplang.v2 as mp
import mplang.v2.edsl.context as ctx_mod
from mplang.v2.dialects import simp
from mplang.v2.dialects.simp import make_simulator
from mplang.v2.edsl.context import get_root_context


class TestContextRoot(unittest.TestCase):
    def setUp(self):
        # Clean context stack
        # Must access via module to handle reference updates by _replace_context_stack
        ctx_mod._context_stack.clear()

    def tearDown(self):
        ctx_mod._context_stack.clear()

    def test_tracer_interaction(self):
        """Verify that Tracers sit on top of Root, keeping Root valid."""
        sim = make_simulator(3)
        with sim:
            # stack=[sim] (Root)
            self.assertIs(get_root_context(), sim)

            @mp.compile
            def job():
                # Inside job: stack=[sim, Tracer]
                # Root should still be sim
                root = get_root_context()
                curr = mp.get_current_context()

                assert root is sim
                assert curr is not sim  # It is a Tracer

                mp.device("P0")
                return simp.constant((0,), 1)

            # Note: @mp.compile executes tracing immediately.

    def test_compile_auto_context(self):
        """Test mp.compile uses the context provided in args or set_root_context."""
        sim = make_simulator(3)

        # Case 1: set_root_context
        mp.set_root_context(sim, force=True)

        captured_root = None

        @mp.compile
        def job1():
            nonlocal captured_root
            captured_root = get_root_context()
            return simp.constant((0,), 1)

        # check side effect immediately
        self.assertIs(captured_root, sim)

        # Case 2: context arg to compile
        sim2 = make_simulator(3)
        captured_cluster = None

        def func():
            nonlocal captured_cluster
            # With the simplified design, the context stack is traversed
            # to find the interpreter with _cluster_spec.
            # sim2 is pushed to the stack by compile(context=sim2).
            from mplang.v2.libs.device.api import _resolve_cluster

            captured_cluster = _resolve_cluster()
            return simp.constant((0,), 1)

        mp.compile(func, context=sim2)
        # compile() executes trace(). sim2's cluster should be resolved.
        self.assertIs(captured_cluster, sim2._cluster_spec)


if __name__ == "__main__":
    unittest.main()
