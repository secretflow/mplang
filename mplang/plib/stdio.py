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

from typing import Any

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.base import MPObject, TensorInfo
from mplang.core.pfunc import PFunction


def read(
    path: str, ty: TensorInfo, **kwargs: Any
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """
    Read data from a file.

    Args:
        path: The file path to read from
        ty: The tensor type information for the data
        **kwargs: Additional attributes for reading

    Returns:
        tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for reading, empty args list, output tree definition
    """
    pfunc = PFunction(
        fn_name="Read",
        fn_type="Read",
        fn_text="",
        fn_body=None,
        ins_info=(),
        outs_info=(ty,),
        attrs={"path": path, **kwargs},
    )
    _, treedef = tree_flatten(ty)
    return pfunc, [], treedef


def write(obj: MPObject, path: str) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """
    Write data to a file.

    Args:
        obj: The MPObject to write
        path: The file path to write to

    Returns:
        tuple[PFunction, list[MPObject], PyTreeDef]: PFunction for writing, args list with obj, output tree definition

    Raises:
        ValueError: If obj is not provided
    """
    if obj is None:
        raise ValueError("stdio.write requires an object to write")

    obj_ty = TensorInfo.from_obj(obj)
    pfunc = PFunction(
        fn_name="Write",
        fn_type="Write",
        fn_text="",
        fn_body=None,
        ins_info=(obj_ty,),
        outs_info=(obj_ty,),
        attrs={"path": path},
    )
    _, treedef = tree_flatten(obj_ty)
    return pfunc, [obj], treedef
