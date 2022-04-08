# Copyright 2021 Google LLC
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
"""RQMC support."""

from tf_quant_finance.math.qmc import utils
from tf_quant_finance.math.qmc.digital_net import digital_net_sample
from tf_quant_finance.math.qmc.digital_net import random_digital_shift
from tf_quant_finance.math.qmc.digital_net import random_scrambling_matrices
from tf_quant_finance.math.qmc.digital_net import scramble_generating_matrices
from tf_quant_finance.math.qmc.lattice_rule import lattice_rule_sample
from tf_quant_finance.math.qmc.lattice_rule import random_scrambling_vectors
from tf_quant_finance.math.qmc.sobol import sobol_generating_matrices
from tf_quant_finance.math.qmc.sobol import sobol_sample

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'digital_net_sample',
    'lattice_rule_sample',
    'random_digital_shift',
    'random_scrambling_matrices',
    'random_scrambling_vectors',
    'scramble_generating_matrices',
    'sobol_generating_matrices',
    'sobol_sample',
    'utils',
]

remove_undocumented(__name__, _allowed_symbols)
