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
"""Optimization methods."""


from tensorflow_probability.python.optimizer import bfgs_minimize
from tensorflow_probability.python.optimizer import converged_all
from tensorflow_probability.python.optimizer import converged_any
from tensorflow_probability.python.optimizer import differential_evolution_minimize
from tensorflow_probability.python.optimizer import differential_evolution_one_step
from tensorflow_probability.python.optimizer import lbfgs_minimize
from tensorflow_probability.python.optimizer import linesearch
from tensorflow_probability.python.optimizer import nelder_mead_minimize
from tensorflow_probability.python.optimizer import nelder_mead_one_step

from tf_quant_finance.math.optimizer.conjugate_gradient import ConjugateGradientParams
from tf_quant_finance.math.optimizer.conjugate_gradient import minimize as conjugate_gradient_minimize
from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'bfgs_minimize',
    'differential_evolution_minimize',
    'differential_evolution_one_step',
    'conjugate_gradient_minimize',
    'converged_all',
    'converged_any',
    'lbfgs_minimize',
    'linesearch',
    'nelder_mead_minimize',
    'nelder_mead_one_step',
    'ConjugateGradientParams',
]

remove_undocumented(__name__, _allowed_symbols)
