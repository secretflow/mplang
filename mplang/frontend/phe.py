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

from mplang.core.dtype import UINT8
from mplang.core.tensor import TensorType
from mplang.frontend.base import stateless_mod

_PHE_MOD = stateless_mod("phe")


@_PHE_MOD.simple_op()
def keygen(
    *, scheme: str = "paillier", key_size: int = 2048
) -> tuple[TensorType, TensorType]:
    """Generate a PHE key pair: returns (public_key, private_key).

    Keys are represented with a sentinel TensorType UINT8[(-1, 0)] to indicate
    non-structural, backend-only handles. Runtime validation will treat this
    shape as an opaque placeholder and skip dtype/shape checks.
    """
    key_spec = TensorType(UINT8, (-1, 0))
    return key_spec, key_spec


@_PHE_MOD.simple_op()
def encrypt(plaintext: TensorType, public_key: TensorType) -> TensorType:
    """Encrypt plaintext using PHE public key: returns ciphertext with same semantic type as plaintext."""
    _ = public_key
    return plaintext


@_PHE_MOD.simple_op()
def add(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Add two PHE operands (semantics depend on backend representation)."""
    _ = operand2
    return operand1


@_PHE_MOD.simple_op()
def mul(ciphertext: TensorType, plaintext: TensorType) -> TensorType:
    """Multiply a PHE ciphertext with a plaintext value (ciphertext dtype preserved)."""
    if ciphertext.dtype.is_floating and plaintext.dtype.is_floating:
        raise ValueError(
            "PHE multiplication does not support float x float operations due to truncation requirements. "
            "Consider using mixed types (float x int) or integer types instead."
        )
    return ciphertext


@_PHE_MOD.simple_op()
def decrypt(ciphertext: TensorType, private_key: TensorType) -> TensorType:
    """Decrypt ciphertext using PHE private key: returns plaintext with same semantic type as ciphertext."""
    _ = private_key
    return ciphertext


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
        self, ciphertext: MPObject, indices: MPObject, axis: int = 0, **kwargs: Any
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Gather elements from ciphertext using indices.

        Args:
            ciphertext: The ciphertext to gather from
            indices: The indices to gather
            axis: The axis along which to gather (default: 0)
            **kwargs: Additional operation parameters

        Returns:
            tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for gather, args list, output tree definition
        """
        ct_ty = TensorType.from_obj(ciphertext)
        indices_ty = TensorType.from_obj(indices)

        # Calculate result shape based on axis parameter
        ct_shape = list(ct_ty.shape)
        indices_shape = list(indices_ty.shape)

        # Normalize negative axis
        normalized_axis = axis if axis >= 0 else len(ct_shape) + axis

        # Result shape: replace the axis dimension with indices shape
        result_shape = (
            ct_shape[:normalized_axis] + indices_shape + ct_shape[normalized_axis + 1 :]
        )
        result_ty = TensorType(ct_ty.dtype, tuple(result_shape))

        pfunc = PFunction(
            fn_type="phe.gather",
            ins_info=(ct_ty, indices_ty),
            outs_info=(result_ty,),
            axis=axis,
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [ciphertext, indices], treedef


gather = Gather()


class Scatter(FEOp):
    """Scatter function class."""

    def __call__(
        self,
        ciphertext: MPObject,
        indices: MPObject,
        updates: MPObject,
        axis: int = 0,
        **kwargs: Any,
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        """Scatter updates into ciphertext at specified indices.

        Args:
            ciphertext: The ciphertext to scatter into
            indices: The indices to scatter at
            updates: The ciphertext updates to scatter
            axis: The axis along which to scatter (default: 0)
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
            axis=axis,
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

        pfunc_kwargs = {}
        if axes is not None:
            pfunc_kwargs["axes"] = axes

        pfunc = PFunction(
            fn_type="phe.transpose",
            ins_info=(ct_ty,),
            outs_info=(result_ty,),
            **pfunc_kwargs,
            **kwargs,
        )
        _, treedef = tree_flatten(result_ty)
        return pfunc, [ciphertext], treedef


transpose = Transpose()
