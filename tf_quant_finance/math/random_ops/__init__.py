# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ops related to random or quasi random sampling."""


from tf_quant_finance.math.random_ops import halton
from tf_quant_finance.math.random_ops import sobol
from tf_quant_finance.math.random_ops.multivariate_normal import multivariate_normal as mv_normal_sample
from tf_quant_finance.math.random_ops.multivariate_normal import RandomType
from tf_quant_finance.math.random_ops.stateless import stateless_random_shuffle
from tf_quant_finance.math.random_ops.uniform import uniform
from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'halton',
    'sobol',
    'mv_normal_sample',
    'RandomType',
    'stateless_random_shuffle',
    'uniform',
]

remove_undocumented(__name__, _allowed_symbols)
