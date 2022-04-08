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

"""LSM algorithm methods."""

from tf_quant_finance.models.longstaff_schwartz.lsm import least_square_mc
from tf_quant_finance.models.longstaff_schwartz.lsm import make_polynomial_basis
from tf_quant_finance.models.longstaff_schwartz.payoff_utils import make_basket_put_payoff

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    "least_square_mc",
    "make_basket_put_payoff",
    "make_polynomial_basis",
]

remove_undocumented(__name__, _allowed_symbols)
