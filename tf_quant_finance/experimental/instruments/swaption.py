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
"""European swaptions."""

import tensorflow.compat.v2 as tf
from tf_quant_finance import black_scholes
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import rates_common as rc


class Swaption:
  """Represents a batch of European Swaptions.

  A European Swaption is a contract that gives the holder an option to enter a
  swap contract at a future date at a prespecified fixed rate. A swaption that
  grants the holder to pay fixed rate and receive floating rate is called a
  payer swaption while the swaption that grants the holder to receive fixed and
  pay floating payments is called the receiver swaption. Typically the start
  date (or the inception date) of the swap concides with the expiry of the
  swaption [1].

  The Swaption class can be used to create and price multiple swaptions
  simultaneously. However all swaptions within an object must be priced using
  common reference/discount curve.

  ### Example:
  The following example illustrates the construction of an European swaption
  and calculating its price using the Black model.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments
  rc = tff.experimental.instruments.rates_common

  dtype = np.float64
  notional = 1.e6
  maturity_date = dates.convert_to_date_tensor([(2025, 2, 8)])
  start_date = dates.convert_to_date_tensor([(2022, 2, 8)])
  expiry_date = dates.convert_to_date_tensor([(2022, 2, 8)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])

  period3m = dates.periods.months(3)
  period6m = dates.periods.months(6)
  fix_spec = instruments.FixedCouponSpecs(
      coupon_frequency=period6m, currency='usd', notional=notional,
      coupon_rate=0.03134,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365,
      businessday_rule=dates.BusinessDayConvention.NONE)
  flt_spec = instruments.FloatCouponSpecs(
      coupon_frequency=period3m, reference_rate_term=period3m,
      reset_frequency=period3m, currency='usd', notional=notional,
      businessday_rule=dates.BusinessDayConvention.NONE,
      coupon_basis=0., coupon_multiplier=1.,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365)

  swap = instruments.InterestRateSwap(start_date, maturity_date,
                                      [fix_spec], [flt_spec],
                                      dtype=dtype)
  swaption = instruments.Swaption(swap, expiry_date, dtype=dtype)

  curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])

  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([
          0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
          0.03213901, 0.03257991
      ], dtype=np.float64),
      valuation_date=valuation_date,
      dtype=np.float64)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = swaption.price(
          valuation_date,
          market,
          model=instruments.InterestRateModelType.LOGNORMAL_RATE,
          pricing_context=0.5))
  # Expected result: 24145.254011
  ```

  ### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

  def __init__(self,
               swap,
               expiry_date=None,
               dtype=None,
               name=None):
    """Initialize a batch of European swaptions.

    Args:
      swap: An instance of `InterestRateSwap` specifying the interest rate
        swaps underlying the swaptions. The batch size of the swaptions being
        created would be the same as the batch size of the `swap`.
      expiry_date: An optional rank 1 `DateTensor` specifying the expiry dates
        for each swaption. The shape of the input should be the same as the
        batch size of the `swap` input.
        Default value: None in which case the option expity date is the same as
        the start date of each underlying swap.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the Swaption object or created by the Swaption
        object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'swaption'.
    """
    self._name = name or 'swaption'

    with tf.name_scope(self._name):
      self._dtype = dtype
      self._expiry_date = dates.convert_to_date_tensor(expiry_date)
      self._swap = swap

  def price(self, valuation_date, market, model=None, pricing_context=None,
            name=None):
    """Returns the present value of the swaption on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the FRA instrument.
      model: An optional input of type `InterestRateModelType` to specify which
        model to use for pricing.
        Default value: `None` in which case LOGNORMAL_RATE model is used.
      pricing_context: An optional input to provide additional parameters (such
        as model parameters) relevant for pricing.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A Rank 1 `Tensor` of real type containing the modeled price of each IRS
      contract based on the input market data.

    Raises:
      ValueError: If an unsupported model is supplied to the function.
    """
    model = model or rc.InterestRateModelType.LOGNORMAL_RATE
    name = name or (self._name + '_price')
    with tf.name_scope(name):
      swap_annuity = self._swap.annuity(valuation_date, market, model)
      forward_swap_rate = self._swap.par_rate(valuation_date, market, model)
      strike = self._swap.fixed_rate

      expiry_time = dates.daycount_actual_365_fixed(
          start_date=valuation_date,
          end_date=self._expiry_date,
          dtype=self._dtype)
      # Ideally we would like the model to tell us how to price the option.
      # The default for European swaptions should be SABR, but the current
      # implementation needs some work.
      if model == rc.InterestRateModelType.LOGNORMAL_RATE:
        option_value = self._price_lognormal_rate(market, pricing_context,
                                                  forward_swap_rate,
                                                  strike, expiry_time)
      else:
        raise ValueError('Unsupported model.')

      return self._swap.notional[-1] * swap_annuity * option_value

  def _price_lognormal_rate(self, market, pricing_context,
                            forward_swap_rate, strike,
                            expiry_time):
    """Price the swaption using lognormal model for rate."""

    # Ideally we would like the model to tell what piece of market data is
    # needed. For example, a Black lognormal model will tell us to pick
    # lognormal vols and Black normal model should tell us to pick normal
    # vols.
    if pricing_context is None:
      swaption_vol_cube = rc.get_implied_volatility_data(market)
      term = self._swap.swap_term
      black_vols = swaption_vol_cube.interpolate(self._expiry_date, strike,
                                                 term)
    else:
      black_vols = tf.convert_to_tensor(pricing_context, dtype=self._dtype)
    return black_scholes.option_price(volatilities=black_vols,
                                      strikes=strike,
                                      expiries=expiry_time,
                                      forwards=forward_swap_rate,
                                      is_call_options=self._swap.is_payer,
                                      dtype=self._dtype
                                      )
