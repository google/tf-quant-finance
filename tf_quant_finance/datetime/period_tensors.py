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
"""PeriodTensor definition."""

from tf_quant_finance.datetime import periods
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


day = deprecation.deprecated_alias(
    'tff.datetime.day',
    'tff.datetime.periods.day',
    periods.day)

days = deprecation.deprecated_alias(
    'tff.datetime.days',
    'tff.datetime.periods.days',
    periods.days)

week = deprecation.deprecated_alias(
    'tff.datetime.week',
    'tff.datetime.periods.week',
    periods.week)

weeks = deprecation.deprecated_alias(
    'tff.datetime.weeks',
    'tff.datetime.periods.weeks',
    periods.weeks)

month = deprecation.deprecated_alias(
    'tff.datetime.month',
    'tff.datetime.periods.month',
    periods.month)

months = deprecation.deprecated_alias(
    'tff.datetime.months',
    'tff.datetime.periods.months',
    periods.months)

year = deprecation.deprecated_alias(
    'tff.datetime.year',
    'tff.datetime.periods.year',
    periods.year)

years = deprecation.deprecated_alias(
    'tff.datetime.years',
    'tff.datetime.periods.years',
    periods.years)


__all__ = [
    'day',
    'days',
    'month',
    'months',
    'week',
    'weeks',
    'year',
    'years',
]
