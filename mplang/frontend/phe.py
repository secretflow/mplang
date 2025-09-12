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


class Dot(FEOp):
    """Dot function class."""

    def __call__(
        self, ciphertext: MPObject, plaintext: MPObject, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Compute dot product of ciphertext with plaintext.

        Args:
            ciphertext: The ciphertext operand (first argument)
            plaintext: The plaintext operand (second argument)
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for dot product, args list, output tree definition
        """
        ct_ty = TensorType.from_obj(ciphertext)
        pt_ty = TensorType.from_obj(plaintext)

        # For dot product, we need to calculate the result shape
        # This follows numpy dot product rules
        import numpy as np

        # Create dummy arrays to determine result shape
        dummy_ct = np.zeros(ct_ty.shape)
        dummy_pt = np.zeros(pt_ty.shape)
        dummy_result = np.dot(dummy_ct, dummy_pt)

        result_ty = TensorType(ct_ty.dtype, dummy_result.shape)

        pfunc = PFunction(
            fn_type="phe.dot",
            ins_info=(ct_ty, pt_ty),
            outs_info=(result_ty,),
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [ciphertext, plaintext], treedef


dot = Dot()


class Gather(FEOp):
    """Gather function class."""

    def __call__(
        self, ciphertext: MPObject, indices: MPObject, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Gather elements from ciphertext using indices.

        Args:
            ciphertext: The ciphertext to gather from
            indices: The indices to gather
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for gather, args list, output tree definition
        """
        ct_ty = TensorType.from_obj(ciphertext)
        indices_ty = TensorType.from_obj(indices)

        # Result shape is same as indices shape, but with ciphertext dtype
        result_ty = TensorType(ct_ty.dtype, indices_ty.shape)

        pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(ct_ty, indices_ty),
            outs_info=(result_ty,),
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [ciphertext, indices], treedef


gather = Gather()


class Scatter(FEOp):
    """Scatter function class."""

    def __call__(
        self, ciphertext: MPObject, indices: MPObject, updates: MPObject, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Scatter updates into ciphertext at specified indices.

        Args:
            ciphertext: The ciphertext to scatter into
            indices: The indices to scatter at
            updates: The ciphertext updates to scatter
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for scatter, args list, output tree definition
        """
        ct_ty = TensorType.from_obj(ciphertext)
        indices_ty = TensorType.from_obj(indices)
        updates_ty = TensorType.from_obj(updates)

        # Result has same shape and dtype as original ciphertext
        result_ty = ct_ty

        pfunc = PFunction(
            fn_type="phe.scatter",
            ins_info=(ct_ty, indices_ty, updates_ty),
            outs_info=(result_ty,),
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [ciphertext, indices, updates], treedef


scatter = Scatter()


class Concat(FEOp):
    """Concat function class."""

    def __call__(
        self, operands: list[MPObject], axis: int = 0, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Concatenate ciphertext tensors along specified axis.

        Args:
            operands: List of ciphertext operands to concatenate
            axis: Axis along which to concatenate
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for concatenation, args list, output tree definition
        """
        if not operands:
            raise ValueError("concat requires at least one operand")

        # Get types of all operands
        operand_types = [TensorType.from_obj(op) for op in operands]

        # All operands should have same dtype
        first_dtype = operand_types[0].dtype
        for op_ty in operand_types[1:]:
            if op_ty.dtype != first_dtype:
                raise ValueError(
                    "All operands must have the same dtype for concatenation"
                )

        # Calculate result shape using numpy concatenation logic
        import numpy as np

        dummy_arrays = [np.zeros(op_ty.shape) for op_ty in operand_types]
        dummy_result = np.concatenate(dummy_arrays, axis=axis)

        result_ty = TensorType(first_dtype, dummy_result.shape)

        pfunc = PFunction(
            fn_type="phe.concat",
            ins_info=tuple(operand_types),
            outs_info=(result_ty,),
            axis=axis,
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, operands, treedef


concat = Concat()


class Reshape(FEOp):
    """Reshape function class."""

    def __call__(
        self, ciphertext: MPObject, new_shape: tuple[int, ...], **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Reshape ciphertext to new shape.

        Args:
            ciphertext: The ciphertext to reshape
            new_shape: The target shape (can contain -1 for inferred dimension)
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for reshape, args list, output tree definition
        """
        ct_ty = TensorType.from_obj(ciphertext)

        # Calculate the actual result shape (handling -1 inference)
        import numpy as np

        dummy_array = np.zeros(ct_ty.shape)
        dummy_result = dummy_array.reshape(new_shape)
        actual_shape = dummy_result.shape

        result_ty = TensorType(ct_ty.dtype, actual_shape)

        pfunc = PFunction(
            fn_type="phe.reshape",
            ins_info=(ct_ty,),
            outs_info=(result_ty,),
            new_shape=new_shape,
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [ciphertext], treedef


reshape = Reshape()


class Transpose(FEOp):
    """Transpose function class."""

    def __call__(
        self, ciphertext: MPObject, axes: tuple[int, ...] | None = None, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Transpose ciphertext by permuting axes.

        Args:
            ciphertext: The ciphertext to transpose
            axes: Permutation of axes (None for default reverse order)
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for transpose, args list, output tree definition
        """
        ct_ty = TensorType.from_obj(ciphertext)

        # Calculate result shape using numpy transpose logic
        import numpy as np

        dummy_array = np.zeros(ct_ty.shape)
        dummy_result = np.transpose(dummy_array, axes)

        result_ty = TensorType(ct_ty.dtype, dummy_result.shape)

        pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(ct_ty,),
            outs_info=(result_ty,),
            axes=axes,
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [ciphertext], treedef


transpose = Transpose()
