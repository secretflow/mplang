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

"""PHE (Partially Homomorphic Encryption) frontend operations."""

from typing import Any

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.dtype import COMPLEX128
from mplang.core.mpobject import MPObject
from mplang.core.mptype import TensorType
from mplang.core.pfunc import PFunction
from mplang.frontend.base import FEOp


class Keygen(FEOp):
    """Keygen function class."""

    def __call__(
        self,
        scheme: str = "paillier",
        key_size: int = 2048,
        **kwargs: Any,
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Generate a PHE key pair.

        Args:
            scheme: The PHE scheme to use ("paillier" or "elgamal")
            key_size: Size of the key in bits
            **kwargs: Additional scheme-specific parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]:
            Returns (public_key, private_key) as two separate outputs.
        """
        # For keys, dtype is meaningless as they shouldn't be used for computation,
        # so we choose a relatively uncommon dtype to satisfy the type system
        public_key_ty = TensorType(COMPLEX128, (1,))
        private_key_ty = TensorType(COMPLEX128, (1,))

        pfunc = PFunction(
            fn_type="phe.keygen",
            ins_info=(),
            outs_info=(public_key_ty, private_key_ty),
            scheme=scheme,
            key_size=key_size,
            **kwargs,
        )

        _, treedef = tree_flatten((public_key_ty, private_key_ty))
        return pfunc, [], treedef


keygen = Keygen()


class Encrypt(FEOp):
    """Encrypt function class."""

    def __call__(
        self, plaintext: MPObject, public_key: MPObject, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Encrypt plaintext using PHE public key.

        Args:
            plaintext: The plaintext tensor to encrypt
            public_key: The PHE public key
            **kwargs: Additional encryption parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]:
        """
        plaintext_ty = TensorType.from_obj(plaintext)
        key_ty = TensorType.from_obj(public_key)

        # Ciphertext has the same semantic type as plaintext (float tensor remains float)
        # but the storage type is backend-dependent
        ciphertext_ty = plaintext_ty

        pfunc = PFunction(
            fn_type="phe.encrypt",
            ins_info=(plaintext_ty, key_ty),
            outs_info=(ciphertext_ty,),
            **kwargs,
        )

        _, treedef = tree_flatten(ciphertext_ty)
        return pfunc, [plaintext, public_key], treedef


encrypt = Encrypt()


class Add(FEOp):
    """Add function class."""

    def __call__(
        self, operand1: MPObject, operand2: MPObject, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Add two PHE operands (can be plaintext or ciphertext).

        Args:
            operand1: First operand (plaintext or ciphertext)
            operand2: Second operand (plaintext or ciphertext)
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for addition, args list, output tree definition

        Note:
            MPObject only represents their semantic type. The actual operands can be:
            - plaintext + plaintext
            - plaintext + ciphertext
            - ciphertext + plaintext
            - ciphertext + ciphertext
        """
        op1_ty = TensorType.from_obj(operand1)
        op2_ty = TensorType.from_obj(operand2)

        # Result has the same semantic type as inputs
        result_ty = op1_ty

        pfunc = PFunction(
            fn_type="phe.add",
            ins_info=(op1_ty, op2_ty),
            outs_info=(result_ty,),
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [operand1, operand2], treedef


add = Add()


class Mul(FEOp):
    """Mul function class."""

    def __call__(
        self, ciphertext: MPObject, plaintext: MPObject, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Multiply a PHE ciphertext with a plaintext value.

        This operation supports multiplication of a ciphertext with a plaintext
        in Paillier encryption scheme, which is homomorphic for multiplication
        with plaintext values.

        Note: This operation does not support floating-point x floating-point multiplication
        due to truncation requirements in the underlying Paillier implementation. However,
        it does support mixed-type operations like floating-point x integer.

        Args:
            ciphertext: The ciphertext to multiply
            plaintext: The plaintext multiplier
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for multiplication, args list, output tree definition

        Raises:
            ValueError: If attempting to multiply two floating-point numbers
        """
        ct_ty = TensorType.from_obj(ciphertext)
        pt_ty = TensorType.from_obj(plaintext)

        # Check if both operands are floating-point numbers
        # PHE multiplication cannot handle float x float due to truncation requirements
        # but can handle float x int or int x float
        if ct_ty.dtype.is_floating and pt_ty.dtype.is_floating:
            raise ValueError(
                "PHE multiplication does not support float x float operations due to truncation requirements. "
                "Consider using mixed types (float x int) or integer types instead."
            )

        # Result has the same semantic type as the ciphertext
        result_ty = ct_ty

        pfunc = PFunction(
            fn_type="phe.mul",
            ins_info=(ct_ty, pt_ty),
            outs_info=(result_ty,),
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [ciphertext, plaintext], treedef


mul = Mul()


class Decrypt(FEOp):
    """Decrypt function class."""

    def __call__(
        self, ciphertext: MPObject, private_key: MPObject, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Decrypt ciphertext using PHE private key.

        Args:
            ciphertext: The ciphertext to decrypt
            private_key: The PHE private key
            **kwargs: Additional decryption parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for decryption, args list, output tree definition
        """
        ciphertext_ty = TensorType.from_obj(ciphertext)
        key_ty = TensorType.from_obj(private_key)

        # Plaintext has the same semantic type as ciphertext, but is no longer encrypted
        plaintext_ty = ciphertext_ty

        pfunc = PFunction(
            fn_type="phe.decrypt",
            ins_info=(ciphertext_ty, key_ty),
            outs_info=(plaintext_ty,),
            **kwargs,
        )
        _, treedef = tree_flatten(plaintext_ty)
        return pfunc, [ciphertext, private_key], treedef


decrypt = Decrypt()
