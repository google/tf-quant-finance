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

"""Futures contracts on overnight rates."""

import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import rates_common as rc


class OvernightIndexLinkedFutures:
  """Represents a collection of futures linked to an average of overnight rates.

  Overnight index futures are exchange traded futures contracts where the
  underlying reference rates are the published overnight rates such as
  Secured Overnight Financing Rate (SOFR), Effective Fed Funds Rate (EFFR) etc.
  These contracts are generally cash settled where the settlement price is
  evaluated on the basis of realized reference rate values during the contract
  reference period (or delivery period). Typically the settlement price is
  based on componding the published daily reference rate during the delivery
  period or based on the arithmetic average of the reference rate during the
  delivery period.
  An overnight index future contract on the settlement date T settles at the
  price

  `100 * (1 - R)`

  If R is evaluated based on compunding the realized index values during the
  reference period then:

  `R = [Product[(1 + tau_i * r_i), 1 <= i <= N] - 1] / Sum[tau_i, 1 <= i <= N]`

  If R is evaluated based on the arithmetic average of the realized index
  during the reference period, then:

  `R = Sum(r_i, 1 <= i <= N)  / N`

  where `i` is the variable indexing the business days within the delivery
  period, tau_i denotes the year fractions between successive business days
  taking into account the appropriate daycount convention and N is the number of
  calendar days in the delivery period. See [1] for SOFR futures on CME.

  The OvernightIndexLinkedFutures class can be used to create and price multiple
  contracts simultaneously. However all contracts within an object must be
  priced using a common reference curve.

  #### Example:
  The following example illustrates the construction of an overnight index
  future instrument and calculating its price.

  ```python

  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff

  dates = tff.datetime
  instruments = tff.experimental.instruments

  dtype = np.float64
  notional = 1.
  contract_start_date = dates.convert_to_date_tensor([(2021, 2, 8)])
  contract_end_date = dates.convert_to_date_tensor([(2021, 5, 8)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])

  future = instruments.OvernightIndexLinkedFutures(
      contract_start_date, contract_end_date, dtype=dtype)

  curve_dates = valuation_date + dates.months([1, 2, 3, 12, 24, 60])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype),
      dtype=dtype)

  market = instruments.InterestRateMarket(reference_curve=reference_curve,
                                          discount_curve=None)

  price = future.price(valuation_date, market)

  #### References:
  [1]: SOFR futures settlement calculation.
  https://www.cmegroup.com/education/files/sofr-futures-settlement-calculation-methodologies.pdf
  """

  def __init__(self,
               contract_start_date,
               contract_end_date,
               daycount_convention=None,
               averaging_type=None,
               contract_unit=1.,
               holiday_calendar=None,
               dtype=None,
               name=None):
    """Initialize the Overnight index futures object.

    Args:
      contract_start_date: A Rank 1 `DateTensor` specifying the start dates of
        the reference period (or delivery period) of each futures contract. The
        published overnight index during the reference period determines the
        final settlement price of the futures contract.
      contract_end_date: A Rank 1 `DateTensor` specifying the ending dates of
        the reference period (or delivery period) of each futures contract.
      daycount_convention: An optional scalar `DayCountConvention` corresponding
        to the day count convention for the underlying rate for each contract.
        Default value: None in which case each the day count convention equal to
        DayCountConvention.ACTUAL_360 is used.
      averaging_type: An optional `AverageType` corresponding to how the
        final settlement rate is computed from daily rates.
        Default value: None, in which case `AverageType.COMPOUNDING` is used.
      contract_unit: An optional scalar or Rank 1 `Tensor` of real dtype
        specifying the notional amount for the contract. If the notional is
        entered as a scalar, it is assumed that all of the contracts have a
        notional equal to the input value.
        Default value: 1.0
      holiday_calendar: An instance of `dates.HolidayCalenday` to specify
        weekends and holidays.
        Default value: None in which case a holiday calendar would be created
        with Saturday and Sunday being the holidays.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the EurodollarFuture object or created by the
        EurodollarFuture object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'eurodollar_future'.
    """
    self._name = name or 'overnight_rate_futures'

    with tf.compat.v2.name_scope(self._name):
      self._contract_unit = tf.convert_to_tensor(
          contract_unit, dtype=dtype)
      self._dtype = dtype if dtype else self._contract_unit.dtype
      self._start_date = dates.convert_to_date_tensor(contract_start_date)
      self._end_date = dates.convert_to_date_tensor(contract_end_date)
      self._batch_size = self._start_date.shape[0]

      if daycount_convention is None:
        daycount_convention = rc.DayCountConvention.ACTUAL_360

      if averaging_type is None:
        averaging_type = rc.AverageType.COMPOUNDING

      if holiday_calendar is None:
        holiday_calendar = dates.create_holiday_calendar(
            weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY)

      self._daycount_convention = daycount_convention
      self._averaging_type = averaging_type
      self._holiday_calendar = holiday_calendar
      self._rate_tenor = dates.day()

      self._setup()

  def price(self, valuation_date, market, model=None, name=None):
    """Returns the price of the contract on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: An object of type `InterestRateMarket` which contains the
        necessary information for pricing the FRA instrument.
      model: Reserved for future use.
      name: Python string. The name to give this op.
        Default value: `None` which maps to `price`.

    Returns:
      A Rank 1 `Tensor` of real type containing the modeled price of each
      futures contract based on the input market data.
    """

    del model, valuation_date

    name = name or (self._name + '_price')
    with tf.name_scope(name):
      reference_curve = market.reference_curve

      df1 = reference_curve.get_discount_factor(self._accrual_start_dates)
      df2 = reference_curve.get_discount_factor(self._accrual_end_dates)

      fwd_rates = (df1 / df2 - 1.) / self._accrual_daycount

      total_accrual = tf.math.segment_sum(self._daycount_fractions,
                                          self._contract_idx)
      if self._averaging_type == rc.AverageType.ARITHMETIC_AVERAGE:

        settlement_rate = tf.math.segment_sum(
            fwd_rates * self._daycount_fractions,
            self._contract_idx) / total_accrual
      else:
        settlement_rate = (tf.math.segment_prod(
            1. + fwd_rates * self._daycount_fractions, self._contract_idx) -
                           1.) / total_accrual

      return 100. * (1. - settlement_rate)

  def _setup(self):
    """Setup relevant tensors for efficient computations."""

    reset_dates = []
    contract_idx = []
    daycount_fractions = []
    for i in range(self._batch_size):
      instr_reset_dates = dates.PeriodicSchedule(
          start_date=self._start_date[i] + self._rate_tenor,
          end_date=self._end_date[i],
          tenor=self._rate_tenor,
          holiday_calendar=self._holiday_calendar,
          roll_convention=dates.BusinessDayConvention.FOLLOWING).dates()

      # Append the start_date of the contract
      instr_reset_dates = dates.DateTensor.concat([
          self._start_date[i].expand_dims(axis=0),
          instr_reset_dates], axis=0)

      # Add one day beyond the end of the delivery period to compute the
      # accrual on the last day of the delivery.
      one_period_past_enddate = self._end_date[i] + self._rate_tenor
      instr_reset_dates = dates.DateTensor.concat([
          instr_reset_dates,
          one_period_past_enddate.expand_dims(axis=0)], axis=0)

      instr_daycount_fractions = rc.get_daycount_fraction(
          instr_reset_dates[:-1], instr_reset_dates[1:],
          self._daycount_convention, self._dtype)

      reset_dates.append(instr_reset_dates[:-1])
      daycount_fractions.append(instr_daycount_fractions)
      contract_idx.append(tf.fill(tf.shape(instr_daycount_fractions), i))

    self._reset_dates = dates.DateTensor.concat(reset_dates, axis=0)
    self._accrual_start_dates = self._reset_dates
    self._accrual_end_dates = self._reset_dates + self._rate_tenor
    self._accrual_daycount = rc.get_daycount_fraction(
        self._accrual_start_dates, self._accrual_end_dates,
        self._daycount_convention, self._dtype)
    self._daycount_fractions = tf.concat(daycount_fractions, axis=0)
    self._contract_idx = tf.concat(contract_idx, axis=0)
