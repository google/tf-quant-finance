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

"""Cap and Floor."""

import tensorflow.compat.v2 as tf
from tf_quant_finance import black_scholes
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import rates_common as rc


class CapAndFloor:
  """Represents a batch of Caps and/or Floors.

  An interest Cap (or Floor) is a portfolio of call (or put) options where the
  underlying for the individual options are successive forward rates. The
  individual options comprising a Cap are called Caplets and the corresponding
  options comprising a Floor are called Floorlets. For example, a
  caplet on forward rate `F(T_i, T_{i+1})` has the following payoff at time
  `T_{i_1}`:

  caplet payoff = tau_i * max[F(T_i, T_{i+1}), 0]

  where `tau_i` is the daycount fraction.

  The CapAndFloor class can be used to create and price multiple Caps/Floors
  simultaneously. However all instruments within an object must be priced using
  common reference/discount curve.

  ### Example:
  The following example illustrates the construction of an IRS instrument and
  calculating its price.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments
  rc = tff.experimental.instruments.rates_common

  dtype = np.float64
  notional = 100.0
  maturity_date = dates.convert_to_date_tensor([(2022, 1, 15)])
  start_date = dates.convert_to_date_tensor([(2021, 1, 15)])
  valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])

  period3m = dates.months(3)
  cap = instruments.CapAndFloor(
      start_date,
      maturity_date,
      period3m,
      0.005,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365,
      notional=notional,
      dtype=dtype)
  curve_dates = valuation_date + dates.months([0, 3, 12, 24])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([0.005, 0.01, 0.015, 0.02], dtype=np.float64),
      valuation_date=valuation_date,
      dtype=np.float64)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = cap.price(
            valuation_date,
            market,
            model=instruments.InterestRateModelType.LOGNORMAL_RATE,
            pricing_context=0.5)
  # Expected result: 1.0474063612452953
  ```

  ### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

  def __init__(self,
               start_date,
               maturity_date,
               reset_frequency,
               strike,
               daycount_convention=None,
               notional=None,
               is_cap=None,
               dtype=None,
               name=None):
    """Initializes a batch of Interest rate Caps (or Floors).

    Args:
      start_date: A scalar `InterestRateSwap` specifying the interest rate swaps
        underlying the swaptions. The batch size of the swaptions being created
        would be the same as the bacth size of the `swap`. For receiver
        swaptions the receive_leg of the underlying swaps should be
      maturity_date: A rank 1 `DateTensor` specifying the expiry dates
        for each swaption. The shape of the input should be the same as the
        batch size of the `swap` input.
        Default value: None in which case the option expity date is the same as
        the start date of each underlying swap.
      reset_frequency: A rank 1 `PeriodTensor` specifying the frequency of
        caplet resets and caplet payments.
      strike: A scalar `Tensor` of real dtype specifying the strike rate against
        which each caplet within the cap are exercised. The shape should be
        compatible to the shape of `start_date`
      daycount_convention: An optional `DayCountConvention` associated with the
        underlying rate for the cap. Daycount is assumed to be the same for all
        contracts in a given batch.
        Default value: None in which case the daycount convention will default
        to DayCountConvention.ACTUAL_360 for all contracts.
      notional: An optional `Tensor` of real dtype specifying the notional
        amount for the cap.
        Default value: None in which case the notional is set to 1.
      is_cap: An optional boolean `Tensor` of a shape compatible with
        `start_date`. Indicates whether to compute the price of a Cap (if True)
        or a Floor (if False).
        Default value: None, it is assumed that every element is a Cap.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the Swaption object or created by the Swaption
        object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'cap_and_floor'.
    """
    self._name = name or 'cap_and_floor'

    with tf.name_scope(self._name):
      self._dtype = dtype
      self._start_date = dates.convert_to_date_tensor(start_date)
      self._maturity_date = dates.convert_to_date_tensor(maturity_date)
      if daycount_convention is None:
        daycount_convention = rc.DayCountConvention.ACTUAL_360
      self._daycount_convention = daycount_convention
      self._strike = tf.convert_to_tensor(strike, dtype=self._dtype)
      self._reset_frequency = reset_frequency
      notional = notional or 1.0
      self._notional = tf.convert_to_tensor(notional, dtype=self._dtype)
      self._batch_size = self._start_date.shape.as_list()[0]
      if is_cap is None:
        is_cap = True
      self._is_cap = tf.broadcast_to(
          tf.convert_to_tensor(is_cap, dtype=tf.bool), self._start_date.shape)
      self._setup_tensors()

  def price(self, valuation_date, market, model=None, pricing_context=None,
            name=None):
    """Returns the present value of the Cap/Floor on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the Cap/Floor.
      model: An optional input of type `InterestRateModelType` to specify which
        model to use for pricing.
        Default value: `None` in which case `LOGNORMAL_RATE` model is used.
      pricing_context: An optional input to provide additional parameters (such
        as model parameters) relevant for pricing.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to `"price"`.

    Returns:
      A Rank 1 `Tensor` of real type containing the modeled price of each cap
      (or floor) based on the input market data.

    Raises:
      ValueError: If an unsupported model is supplied to the function.
    """

    model = model or rc.InterestRateModelType.LOGNORMAL_RATE
    name = name or (self._name + '_price')
    with tf.name_scope(name):
      valuation_date = dates.convert_to_date_tensor(valuation_date)
      if model == rc.InterestRateModelType.LOGNORMAL_RATE:
        caplet_prices = self._price_lognormal_rate(valuation_date, market,
                                                   pricing_context)
      else:
        raise ValueError(f'Unsupported model {model}.')

      return tf.math.segment_sum(caplet_prices, self._contract_index)

  def _price_lognormal_rate(self, valuation_date, market, pricing_context):
    """Computes caplet/floorlet prices using lognormal model for forward rates.

    The function computes individual caplet prices for the batch of caps/floors
    using the lognormal model for the forward rates. If the volatilities are
    are supplied (through the input `pricing_context`) then they are used as
    forward rate volatilies. Otherwise, volatilities are extracted using the
    volatility surface for `market`.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the Cap/Floor.
      pricing_context: An optional input containing the black volatility for
        for the forward rates.

    Returns:
      A Rank 1 `Tensor` of real type containing the price of each caplet
      (or floorlet) based using the lognormal model for forward rates.
    """

    discount_curve = market.discount_curve

    discount_factors = tf.where(
        self._payment_dates > valuation_date,
        discount_curve.get_discount_factor(self._payment_dates), 0.)

    forward_rates = self._get_forward_rate(valuation_date, market)

    if pricing_context is None:
      volatility_surface = market.volatility_curve
      black_vols = volatility_surface.interpolate(
          self._reset_dates, self._strike, self._term)
    else:
      black_vols = tf.convert_to_tensor(pricing_context, dtype=self._dtype)

    expiry_times = dates.daycount_actual_365_fixed(
        start_date=valuation_date, end_date=self._reset_dates,
        dtype=self._dtype)
    caplet_prices = black_scholes.option_price(forwards=forward_rates,
                                               strikes=self._strike,
                                               volatilities=black_vols,
                                               expiries=expiry_times,
                                               is_call_options=self._is_cap)
    intrinsic_value = tf.where(
        self._is_cap, tf.math.maximum(forward_rates - self._strike, 0.0),
        tf.math.maximum(self._strike - forward_rates, 0))
    caplet_prices = tf.where(
        self._payment_dates < valuation_date,
        tf.constant(0., dtype=self._dtype),
        tf.where(self._accrual_start_dates < valuation_date, intrinsic_value,
                 caplet_prices))
    caplet_prices = self._notional * self._daycount_fractions * caplet_prices
    return discount_factors * caplet_prices

  def _setup_tensors(self):
    """Sets up tensors for efficient computations."""
    date_schedule = dates.PeriodicSchedule(
        start_date=self._start_date,
        end_date=self._maturity_date,
        tenor=self._reset_frequency).dates()

    # rates reset at the begining of coupon period
    reset_dates = date_schedule[:, :-1]
    # payments occur at the end of the coupon period
    payment_dates = date_schedule[:, 1:]
    daycount_fractions = rc.get_daycount_fraction(
        date_schedule[:, :-1],
        date_schedule[:, 1:],
        self._daycount_convention,
        dtype=self._dtype)
    contract_index = tf.repeat(
        tf.range(0, self._batch_size),
        payment_dates.shape.as_list()[-1])

    self._num_caplets = daycount_fractions.shape.as_list()[-1]
    # TODO(b/152164086): Use the functionality from dates library
    self._rate_term = tf.repeat(tf.cast(reset_dates[:, 0].days_until(
        payment_dates[:, 0]), dtype=self._dtype) / 365.0, self._num_caplets)
    self._reset_dates = dates.DateTensor.reshape(reset_dates, [-1])
    self._payment_dates = dates.DateTensor.reshape(payment_dates, [-1])
    self._accrual_start_dates = dates.DateTensor.reshape(reset_dates, [-1])
    self._accrual_end_dates = dates.DateTensor.reshape(payment_dates, [-1])
    self._daycount_fractions = tf.reshape(daycount_fractions, [-1])
    self._contract_index = contract_index
    self._strike = tf.repeat(self._strike, self._num_caplets)
    self._is_cap = tf.repeat(self._is_cap, self._num_caplets)

  def _get_forward_rate(self, valuation_date, market):
    """Returns the relevant forward rates from the market data."""

    forward_rates = market.reference_curve.get_forward_rate(
        self._accrual_start_dates,
        self._accrual_end_dates,
        self._daycount_fractions)

    forward_rates = tf.where(self._daycount_fractions > 0.0, forward_rates,
                             tf.zeros_like(forward_rates))

    libor_rate = rc.get_rate_index(
        market, self._start_date, rc.RateIndexType.LIBOR, dtype=self._dtype)
    libor_rate = tf.repeat(
        tf.convert_to_tensor(libor_rate, dtype=self._dtype), self._num_caplets)
    forward_rates = tf.where(
        self._accrual_end_dates < valuation_date,
        tf.constant(0., dtype=self._dtype),
        tf.where(self._accrual_start_dates < valuation_date, libor_rate,
                 forward_rates))

    return forward_rates
