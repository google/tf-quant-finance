# Copyright 2020 Google LLC
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
"""TensorFlow Quantitative Finance tools to build Hull White type models."""

from tf_quant_finance.models.hull_white.calibration import calibration_from_cap_floors
from tf_quant_finance.models.hull_white.calibration import calibration_from_swaptions
from tf_quant_finance.models.hull_white.calibration import CalibrationResult
from tf_quant_finance.models.hull_white.cap_floor import cap_floor_price
from tf_quant_finance.models.hull_white.one_factor import HullWhiteModel1F
from tf_quant_finance.models.hull_white.swaption import bermudan_swaption_price
from tf_quant_finance.models.hull_white.swaption import swaption_price
from tf_quant_finance.models.hull_white.vector_hull_white import VectorHullWhiteModel
from tf_quant_finance.models.hull_white.zero_coupon_bond_option import bond_option_price

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'CalibrationResult',
    'HullWhiteModel1F',
    'VectorHullWhiteModel',
    'bermudan_swaption_price',
    'bond_option_price',
    'cap_floor_price',
    'swaption_price',
    'calibration_from_swaptions',
    'calibration_from_cap_floors',
]

remove_undocumented(__name__, _allowed_symbols)
