# Copyright 2022 Google LLC
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
"""Date periods."""

from tf_quant_finance.datetime.periods.period_tensors import day
from tf_quant_finance.datetime.periods.period_tensors import days
from tf_quant_finance.datetime.periods.period_tensors import month
from tf_quant_finance.datetime.periods.period_tensors import months
from tf_quant_finance.datetime.periods.period_tensors import PeriodTensor
from tf_quant_finance.datetime.periods.period_tensors import week
from tf_quant_finance.datetime.periods.period_tensors import weeks
from tf_quant_finance.datetime.periods.period_tensors import year
from tf_quant_finance.datetime.periods.period_tensors import years

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import


_allowed_symbols = [
    'day',
    'days',
    'week',
    'weeks',
    'month',
    'months',
    'year',
    'years',
    'PeriodTensor',
]

remove_undocumented(__name__, _allowed_symbols)
