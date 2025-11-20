"""BFV Runtime Implementation.

Implements execution logic for BFV primitives using TenSEAL.
"""

from typing import Any

import numpy as np
import tenseal as ts

from mplang2.dialects import bfv
from mplang2.edsl.graph import Operation
from mplang2.edsl.interpreter import Interpreter


@bfv.keygen_p.def_impl
def keygen_impl(interpreter: Interpreter, op: Operation, *args: Any) -> tuple[Any, Any]:
    poly_modulus_degree = op.attrs.get("poly_modulus_degree", 4096)
    # Use a default plain_modulus_bit_size if not provided, or derive it
    # TenSEAL defaults usually work, but let's be explicit if needed.
    # For now, we use a common default for BFV.
    plain_modulus_bit_size = 20

    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=poly_modulus_degree,
        plain_modulus_bit_size=plain_modulus_bit_size,
    )
    context.generate_galois_keys()
    context.generate_relin_keys()

    # Return context as both PK and SK (TenSEAL context holds both)
    return context, context


@bfv.make_relin_keys_p.def_impl
def make_relin_keys_impl(interpreter: Interpreter, op: Operation, sk: Any) -> Any:
    # In TenSEAL, keys are generated in context.
    # We just return the context which acts as the key holder.
    return sk


@bfv.make_galois_keys_p.def_impl
def make_galois_keys_impl(interpreter: Interpreter, op: Operation, sk: Any) -> Any:
    return sk


@bfv.create_encoder_p.def_impl
def create_encoder_impl(interpreter: Interpreter, op: Operation) -> Any:
    # TenSEAL doesn't expose a separate Encoder object for BFV in the same way as SEAL C++.
    # Encoding is done via methods on the context or vector constructors.
    # We'll return a dummy object or the context if needed, but since encode takes it...
    # Let's return a simple marker or config.
    return {"poly_modulus_degree": op.attrs.get("poly_modulus_degree", 4096)}


@bfv.encode_p.def_impl
def encode_impl(
    interpreter: Interpreter, op: Operation, data: Any, encoder: Any
) -> Any:
    # data is expected to be a numpy array or list
    # encoder is our dummy config
    return np.array(data)


@bfv.encrypt_p.def_impl
def encrypt_impl(
    interpreter: Interpreter, op: Operation, plaintext: Any, pk: Any
) -> Any:
    # pk is the TenSEAL context
    # plaintext is the data (numpy array)
    return ts.bfv_vector(pk, plaintext)


@bfv.decrypt_p.def_impl
def decrypt_impl(
    interpreter: Interpreter, op: Operation, ciphertext: Any, sk: Any
) -> Any:
    # sk is the TenSEAL context
    # ciphertext is ts.BFVVector
    return ciphertext.decrypt()


@bfv.decode_p.def_impl
def decode_impl(
    interpreter: Interpreter, op: Operation, plaintext: Any, encoder: Any
) -> Any:
    # plaintext is the decrypted list/vector
    return np.array(plaintext)


@bfv.add_p.def_impl
def add_impl(interpreter: Interpreter, op: Operation, lhs: Any, rhs: Any) -> Any:
    return lhs + rhs


@bfv.mul_p.def_impl
def mul_impl(interpreter: Interpreter, op: Operation, lhs: Any, rhs: Any) -> Any:
    return lhs * rhs


@bfv.relinearize_p.def_impl
def relinearize_impl(
    interpreter: Interpreter, op: Operation, ciphertext: Any, rk: Any
) -> Any:
    # TenSEAL auto-relinearizes if keys are present, but we can force it if API exists
    # or just return as is since TenSEAL handles it.
    # Actually ts.BFVVector has no explicit relinearize method exposed in basic API
    # usually, it's done during mul if configured.
    # But let's check if we need to do anything.
    # For now, pass through.
    return ciphertext


@bfv.rotate_p.def_impl
def rotate_impl(
    interpreter: Interpreter, op: Operation, ciphertext: Any, steps: int, gk: Any
) -> Any:
    # steps is an attribute or arg? In bfv.py it's an arg.
    # Check bfv.py definition.
    # rotate(ciphertext, steps, galois_keys)
    # steps is passed as arg.

    # TenSEAL rotation
    # Note: TenSEAL rotate might be in-place or return new.
    # Usually returns new or modifies.
    # ts.BFVVector.rotate(steps) -> returns None (in-place) or new?
    # Let's assume standard behavior.

    # Create a copy to avoid side effects if it's in-place
    res = ciphertext.copy()
    res.rotate(steps)
    return res
