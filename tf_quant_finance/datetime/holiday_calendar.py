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
"""HolidayCalendar definition."""

import abc

from tf_quant_finance.datetime import constants


class HolidayCalendar(abc.ABC):
  """Represents a holiday calendar.

  Provides methods for manipulating the dates taking into account the holidays,
  and the business day roll conventions. Weekends are treated as holidays.
  """

  @abc.abstractmethod
  def is_business_day(self, date_tensor):
    """Returns a tensor of bools for whether given dates are business days."""
    pass

  @abc.abstractmethod
  def roll_to_business_day(self, date_tensor, roll_convention):
    """Rolls the given dates to business dates according to given convention.

    Args:
      date_tensor: DateTensor of dates to roll from.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    pass

  @abc.abstractmethod
  def add_period_and_roll(self,
                          date_tensor,
                          period_tensor,
                          roll_convention=constants.BusinessDayConvention.NONE):
    """Adds given periods to given dates and rolls to business days.

    The original dates are not rolled prior to addition.

    Args:
      date_tensor: DateTensor of dates to add to.
      period_tensor: PeriodTensor broadcastable to `date_tensor`.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    pass

  @abc.abstractmethod
  def add_business_days(self,
                        date_tensor,
                        num_days,
                        roll_convention=constants.BusinessDayConvention.NONE):
    """Adds given number of business days to given dates.

    Note that this is different from calling `add_period_and_roll` with
    PeriodType.DAY. For example, adding 5 business days to Monday gives the next
    Monday (unless there are holidays on this week or next Monday). Adding 5
    days and rolling means landing on Saturday and then rolling either to next
    Monday or to Friday of the same week, depending on the roll convention.

    If any of the dates in `date_tensor` are not business days, they will be
    rolled to business days before doing the addition. If `roll_convention` is
    `NONE`, and any dates are not business days, an exception is raised.

    Args:
      date_tensor: DateTensor of dates to advance from.
      num_days: Tensor of int32 type broadcastable to `date_tensor`.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    pass

  @abc.abstractmethod
  def subtract_period_and_roll(
      self,
      date_tensor,
      period_tensor,
      roll_convention=constants.BusinessDayConvention.NONE):
    """Subtracts given periods from given dates and rolls to business days.

    The original dates are not rolled prior to subtraction.

    Args:
      date_tensor: DateTensor of dates to subtract from.
      period_tensor: PeriodTensor broadcastable to `date_tensor`.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    pass

  @abc.abstractmethod
  def subtract_business_days(
      self,
      date_tensor,
      num_days,
      roll_convention=constants.BusinessDayConvention.NONE):
    """Adds given number of business days to given dates.

    Note that this is different from calling `subtract_period_and_roll` with
    PeriodType.DAY. For example, subtracting 5 business days from Friday gives
    the previous Friday (unless there are holidays on this week or previous
    Friday). Subtracting 5 days and rolling means landing on Sunday and then
    rolling either to Monday or to Friday, depending on the roll convention.

    If any of the dates in `date_tensor` are not business days, they will be
    rolled to business days before doing the subtraction. If `roll_convention`
    is `NONE`, and any dates are not business days, an exception is raised.

    Args:
      date_tensor: DateTensor of dates to advance from.
      num_days: Tensor of int32 type broadcastable to `date_tensor`.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    pass

  @abc.abstractmethod
  def business_days_in_period(self, date_tensor, period_tensor):
    """Calculates number of business days in a period.

    Includes the dates in `date_tensor`, but excludes final dates resulting from
    addition of `period_tensor`.

    Args:
      date_tensor: DateTensor of starting dates.
      period_tensor: PeriodTensor, should be broadcastable to `date_tensor`.

    Returns:
       An int32 Tensor with the number of business days in given periods that
       start at given dates.

    """
    pass

  @abc.abstractmethod
  def business_days_between(self, from_dates, to_dates):
    """Calculates number of business between pairs of dates.

    For each pair, the initial date is included in the difference, and the final
    date is excluded. If the final date is the same or earlier than the initial
    date, zero is returned.

    Args:
      from_dates: DateTensor of initial dates.
      to_dates: DateTensor of final dates, should be broadcastable to
        `from_dates`.

    Returns:
       An int32 Tensor with the number of business days between the
       corresponding pairs of dates.
    """
    pass
