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
"""Collection of functions to compute properties of cashflows."""

from tf_quant_finance.rates.analytics import cashflows
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

pv_from_yields = deprecation.deprecated_alias(
    'tff.rates.cashflows.pv_from_yields',
    'tff.rates.analytics.cashflows.pv_from_yields',
    cashflows.pv_from_yields)

yields_from_pv = deprecation.deprecated_alias(
    'tff.rates.cashflows.yields_from_pv',
    'tff.rates.analytics.cashflows.yields_from_pv',
    cashflows.yields_from_pv)


__all__ = ['pv_from_yields', 'yields_from_pv']
