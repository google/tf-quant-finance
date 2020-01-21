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
"""Date-related utilities."""

from tf_quant_finance.experimental.dates.constants import BusinessDayConvention
from tf_quant_finance.experimental.dates.constants import Month
from tf_quant_finance.experimental.dates.constants import PeriodType
from tf_quant_finance.experimental.dates.constants import WeekDay
from tf_quant_finance.experimental.dates.date_tensor import DateTensor
import tf_quant_finance.experimental.dates.date_utils
from tf_quant_finance.experimental.dates.holiday_calendar import HolidayCalendar
import tf_quant_finance.experimental.dates.periods
from tf_quant_finance.experimental.dates.periods import PeriodTensor
from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import


_allowed_symbols = [
    'DateTensor',
    'PeriodTensor',
    'PeriodType',
    'periods',
    'Month',
    'WeekDay',
    'HolidayCalendar',
    'BusinessDayConvention',
    'date_utils',
]

remove_undocumented(__name__, _allowed_symbols)
