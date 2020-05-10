# Lint as: python3
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
"""Date-related constants and enums."""

import enum


class Month(enum.Enum):
  """Months. Values are one-based."""
  JANUARY = 1
  FEBUARY = 2
  MARCH = 3
  APRIL = 4
  MAY = 5
  JUNE = 6
  JULY = 7
  AUGUST = 8
  SEPTEMBER = 9
  OCTOBER = 10
  NOVEMBER = 11
  DECEMBER = 12


class WeekDay(enum.Enum):
  """Named days of the week. Values are zero-based with Monday = 0."""
  # We follow Python datetime convention of starting from 0.
  MONDAY = 0
  TUESDAY = 1
  WEDNESDAY = 2
  THURSDAY = 3
  FRIDAY = 4
  SATURDAY = 5
  SUNDAY = 6


class PeriodType(enum.Enum):
  """Periods that can be added or subtracted from DateTensors."""
  DAY = 0
  WEEK = 1
  MONTH = 2
  YEAR = 3


class BusinessDayConvention(enum.Enum):
  """Conventions that determine how to roll dates that fall on holidays.

  * `NONE`: No adjustment
  * `FOLLOWING`: Choose the first business day after the given holiday.
  * `MODIFIED_FOLLOWING`: Choose the first business day after the given holiday
  unless that day falls in the next calendar month, in which case choose the
  first business day before the holiday.
  * `PRECEDING`: Choose the first business day before the given holiday.
  * `MODIFIED_PRECEDING`: Choose the first business day before the given holiday
  unless that day falls in the previous calendar month, in which case choose the
  first business day after the holiday.
  """
  NONE = 0
  FOLLOWING = 1
  MODIFIED_FOLLOWING = 2
  PRECEDING = 3
  MODIFIED_PRECEDING = 4

# TODO(b/148011715): add NEAREST convention.


class WeekendMask(object):
  """Provides weekend masks for some of the common weekend patterns."""

  # E.g. US/UK/Europe etc.
  SATURDAY_SUNDAY = (0, 0, 0, 0, 0, 1, 1)

  # E.g. Most countries in the Middle East.
  FRIDAY_SATURDAY = (0, 0, 0, 0, 1, 1, 0)

  # E.g. India, Nepal.
  SUNDAY_ONLY = (0, 0, 0, 0, 0, 0, 1)

  # Default value.
  NONE = (0, 0, 0, 0, 0, 0, 0)
