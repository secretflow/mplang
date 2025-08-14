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


def identity(x: MPObject) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    io_info = (TensorInfo.from_obj(x),)
    pfunc = PFunction(
        fn_name="Identity",
        fn_type="Identity",
        fn_text="",
        fn_body=None,
        ins_info=io_info,
        outs_info=io_info,
        attrs={},
    )
    _, treedef = tree_flatten(io_info)
    return pfunc, [x], treedef
