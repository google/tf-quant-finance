# Lint as: python3
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

from tf_quant_finance.experimental.rqmc import utils
from tf_quant_finance.experimental.rqmc.digital_net import random_scrambling_matrices
from tf_quant_finance.experimental.rqmc.digital_net import sample_digital_net
from tf_quant_finance.experimental.rqmc.digital_net import scramble_generating_matrices
from tf_quant_finance.experimental.rqmc.lattice_rule import random_scrambling_vectors
from tf_quant_finance.experimental.rqmc.lattice_rule import sample_lattice_rule
from tf_quant_finance.experimental.rqmc.sobol import sample_sobol
from tf_quant_finance.experimental.rqmc.sobol import sobol_generating_matrices

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'random_scrambling_matrices',
    'random_scrambling_vectors',
    'sample_digital_net',
    'sample_lattice_rule',
    'sample_sobol',
    'scramble_generating_matrices',
    'sobol_generating_matrices',
    'utils',
]

remove_undocumented(__name__, _allowed_symbols)
