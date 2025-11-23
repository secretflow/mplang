"""Tests for OT library."""

import mplang2.backends.tensor_impl  # noqa: F401
from mplang2.backends.simp_host import HostVar
from mplang2.backends.simp_simulator import SimpSimulator, get_or_create_context
from mplang2.edsl.interpreter import InterpObject
from mplang2.edsl.typing import MPType, i64
from mplang2.libs import ot


def test_ot_transfer_basic():
    # World size 2: Party 0 (Sender), Party 1 (Receiver)
    interp = SimpSimulator(world_size=2)
    get_or_create_context(2)

    # Sender inputs: m0=10, m1=20
    # Receiver input: choice=1 (should get 20)

    def protocol(m0, m1, choice):
        # This function describes the whole protocol
        # m0, m1 are on Party 0
        # choice is on Party 1

        # We call the library function
        return ot.transfer(m0, m1, choice, sender=0, receiver=1)

    # Setup inputs
    # m0, m1 on Party 0
    m0_val = HostVar([10, None])  # Party 0 has 10
    m1_val = HostVar([20, None])

    # choice on Party 1
    choice_val = HostVar([None, 1])  # Party 1 has 1

    m0_obj = InterpObject(m0_val, MPType(i64, (0,)), interp)
    m1_obj = InterpObject(m1_val, MPType(i64, (0,)), interp)
    choice_obj = InterpObject(choice_val, MPType(i64, (1,)), interp)

    with interp:
        # In the library approach, `ot.transfer` internally uses `pcall` and `shuffle`.
        # So we can call it directly if we are in a tracing context, or if we are just running it.
        # However, `ot.transfer` expects `el.Object`s.
        # Since we are running in eager mode (Simulator), we can call it directly with InterpObjects.

        res = protocol(m0_obj, m1_obj, choice_obj)

    # Check result
    # res should be MPType(i64, (1,))
    # Party 1 should have 20.
    assert res.runtime_obj.values[1] == 20


def test_ot_transfer_choice_0():
    # World size 2: Party 0 (Sender), Party 1 (Receiver)
    interp = SimpSimulator(world_size=2)
    get_or_create_context(2)

    def protocol(m0, m1, choice):
        return ot.transfer(m0, m1, choice, sender=0, receiver=1)

    m0_val = HostVar([10, None])
    m1_val = HostVar([20, None])
    choice_val = HostVar([None, 0])  # Choice 0

    m0_obj = InterpObject(m0_val, MPType(i64, (0,)), interp)
    m1_obj = InterpObject(m1_val, MPType(i64, (0,)), interp)
    choice_obj = InterpObject(choice_val, MPType(i64, (1,)), interp)

    with interp:
        res = protocol(m0_obj, m1_obj, choice_obj)

    assert res.runtime_obj.values[1] == 10
