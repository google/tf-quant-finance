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
"""Utils to manipulate holidays."""

import tensorflow.compat.v2 as tf

# In Gregorian Calendar, 1-Jan-1 was a Monday, hence ordinal 0 corresponds
# to a Sunday.
_DAYOFWEEK_0 = 6


def business_day_mappers(weekend_mask=None, holidays=None):
  """Returns functions to map from ordinal to biz day and back."""
  if weekend_mask is None and holidays is None:
    return (lambda x: (x, tf.ones_like(x, dtype=tf.bool))), (lambda x: x)

  weekday_fwd, weekday_back = _week_day_mappers(weekend_mask)

  if holidays is None:
    return weekday_fwd, weekday_back

  # Apply the weekend adjustment to the holidays as well
  holidays_raw = tf.convert_to_tensor(holidays, dtype=tf.int32)
  holidays, is_weekday = weekday_fwd(holidays_raw)

  # Keep only the holidays that are not on weekends
  holidays = holidays[is_weekday]

  # The above step can lead to an empty holidays set which causes problems.
  # To mitigate this, we add a safe fake holiday.
  holidays = tf.concat([[0], holidays], axis=0)
  reverse_holidays = tf.reverse(-holidays, axis=[0])
  num_holidays = tf.size(holidays) - 1

  def bizday_fwd(x):
    """Calculates business day ordinal and whether it is a business day."""
    left = tf.searchsorted(holidays, x, side='left')
    right = num_holidays - tf.searchsorted(reverse_holidays, -x, side='left')
    is_bizday = tf.not_equal(left, right)
    bizday_ordinal = x - right
    return bizday_ordinal, is_bizday

  cum_holidays = tf.range(num_holidays + 1, dtype=holidays.dtype)
  bizday_at_holidays = holidays - cum_holidays

  def bizday_back(x):
    left = tf.searchsorted(bizday_at_holidays, x, side='left')
    ordinal = x + left - 1
    return ordinal

  def from_ordinal(ordinals):
    """Maps ordinals to business day and whether it is a work day."""
    ordinals = tf.convert_to_tensor(ordinals, dtype=tf.int32)
    weekday_values, is_weekday = weekday_fwd(ordinals)
    biz_ordinal, is_bizday = bizday_fwd(weekday_values)
    return biz_ordinal, (is_weekday & is_bizday)

  def to_ordinal(biz_values):
    """Maps from business day count to ordinals."""
    return weekday_back(bizday_back(biz_values))

  return from_ordinal, to_ordinal


def _week_day_mappers(weekend_mask):
  """Creates functions to map from ordinals to week days and inverse.

  Creates functions to map from ordinal space (i.e. days since 31 Dec 0) to
  week days. The function assigns the value of 0 to the first non weekend
  day in the week starting on Sunday, 31 Dec 1 through to Saturday, 6 Jan 1 and
  the value assigned to each successive work day is incremented by 1. For a day
  that is not a week day, this count is not incremented from the previous week
  day (hence, multiple ordinal days may have the same week day value).

  Args:
    weekend_mask: A bool `Tensor` of length 7 or None. The weekend mask.

  Returns:
    A tuple of callables.
      `forward`: Takes one `Tensor` argument containing ordinals and returns a
        tuple of two `Tensor`s of the same shape as the input. The first
        `Tensor` is of type `int32` and contains the week day value. The second
        is a bool `Tensor` indicating whether the supplied ordinal was a weekend
        day (i.e. True where the day is a weekend day and False otherwise).
      `backward`: Takes one int32 `Tensor` argument containing week day values
        and returns an int32 `Tensor` containing ordinals for those week days.
  """
  if weekend_mask is None:
    default_forward = lambda x: (x, tf.zeros_like(x, dtype=tf.bool))
    identity = lambda x: x
    return default_forward, identity
  weekend_mask = tf.convert_to_tensor(weekend_mask, dtype=tf.bool)
  weekend_mask = tf.roll(weekend_mask, -_DAYOFWEEK_0, axis=0)
  weekday_mask = tf.logical_not(weekend_mask)
  weekday_offsets = tf.cumsum(tf.cast(weekday_mask, dtype=tf.int32))
  num_workdays = weekday_offsets[-1]
  weekday_offsets -= 1  # Adjust the first workday to index 0.
  ordinal_offsets = tf.convert_to_tensor([0, 1, 2, 3, 4, 5, 6], dtype=tf.int32)
  ordinal_offsets = ordinal_offsets[weekday_mask]

  def forward(ordinals):
    """Adjusts the ordinals by removing the number of weekend days so far."""
    mod, remainder = ordinals // 7, ordinals % 7
    weekday_values = mod * num_workdays + tf.gather(weekday_offsets, remainder)
    is_weekday = tf.gather(weekday_mask, remainder)
    return weekday_values, is_weekday

  def backward(weekday_values):
    """Converts from weekend adjusted values to ordinals."""
    return ((weekday_values // num_workdays) * 7 +
            tf.gather(ordinal_offsets, weekday_values % num_workdays))

  return forward, backward
