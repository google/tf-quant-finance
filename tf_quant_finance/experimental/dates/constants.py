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

from enum import Enum


class Month(Enum):
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


class WeekDay(Enum):
  # We follow Python datetime convention of starting from 0.
  MONDAY = 0
  TUESDAY = 1
  WEDNESDAY = 2
  THURSDAY = 3
  FRIDAY = 4
  SATURDAY = 5
  SUNDAY = 6


class PeriodType(Enum):
  DAY = 0
  WEEK = 1
  MONTH = 2
  YEAR = 3
