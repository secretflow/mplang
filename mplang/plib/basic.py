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


from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.base import MPObject, TensorInfo
from mplang.core.pfunc import PFunction


def identity(obj: MPObject) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    obj_ty = TensorInfo.from_obj(obj)
    pfunc = PFunction(
        fn_type="builtin.identity",
        ins_info=(obj_ty,),
        outs_info=(obj_ty,),
        fn_name="Identity",
        fn_text="",
        attrs={},
    )
    _, treedef = tree_flatten(obj_ty)
    return pfunc, [obj], treedef
