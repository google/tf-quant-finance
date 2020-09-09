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
"""TensorFlow Quantitative Finance general math functions."""


from tf_quant_finance.math import integration
from tf_quant_finance.math import interpolation
from tf_quant_finance.math import optimizer
from tf_quant_finance.math import pad
from tf_quant_finance.math import pde
from tf_quant_finance.math import piecewise
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.math import root_search
from tf_quant_finance.math import segment_ops
from tf_quant_finance.math.diff_ops import diff
from tf_quant_finance.math.gradient import fwd_gradient
from tf_quant_finance.math.gradient import gradients
from tf_quant_finance.math.gradient import make_val_and_grad_fn
from tf_quant_finance.math.gradient import value_and_gradient
from tf_quant_finance.math.jacobian import jacobian
from tf_quant_finance.math.jacobian import value_and_jacobian

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'fwd_gradient',
    'gradients',
    'integration',
    'interpolation',
    'optimizer',
    'pad',
    'pde',
    'piecewise',
    'random',
    'root_search',
    'diff',
    'segment_ops',
    'value_and_gradient',
    'make_val_and_grad_fn',
    'jacobian',
    'value_and_jacobian',
]

remove_undocumented(__name__, _allowed_symbols)
