# Lint as: python3
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

from tf_quant_finance.experimental.dates import date_utils
from tf_quant_finance.experimental.dates import daycounts
from tf_quant_finance.experimental.dates import periods

from tf_quant_finance.experimental.dates.constants import BusinessDayConvention
from tf_quant_finance.experimental.dates.constants import Month
from tf_quant_finance.experimental.dates.constants import PeriodType
from tf_quant_finance.experimental.dates.constants import WeekDay
from tf_quant_finance.experimental.dates.constants import WeekendMask

from tf_quant_finance.experimental.dates.date_tensor import convert_to_date_tensor
from tf_quant_finance.experimental.dates.date_tensor import DateTensor
from tf_quant_finance.experimental.dates.date_tensor import from_datetimes
from tf_quant_finance.experimental.dates.date_tensor import from_np_datetimes
from tf_quant_finance.experimental.dates.date_tensor import from_ordinals
from tf_quant_finance.experimental.dates.date_tensor import from_tuples
from tf_quant_finance.experimental.dates.date_tensor import from_year_month_day
from tf_quant_finance.experimental.dates.date_tensor import random_dates

from tf_quant_finance.experimental.dates.holiday_calendar import HolidayCalendar
from tf_quant_finance.experimental.dates.holiday_calendar_v2 import HolidayCalendar as HolidayCalendar2

from tf_quant_finance.experimental.dates.schedules import BusinessDaySchedule
from tf_quant_finance.experimental.dates.schedules import PeriodicSchedule

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import


_allowed_symbols = [
    'BusinessDayConvention',
    'BusinessDaySchedule',
    'DateTensor',
    'HolidayCalendar',
    'HolidayCalendar2',
    'Month',
    'PeriodType',
    'WeekDay',
    'WeekendMask',
    'convert_to_date_tensor',
    'from_datetimes',
    'from_np_datetimes',
    'from_ordinals',
    'from_tuples',
    'from_year_month_day',
    'date_utils',
    'daycounts',
    'PeriodicSchedule',
    'periods',
    'random_dates',
]

remove_undocumented(__name__, _allowed_symbols)
