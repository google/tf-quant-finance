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
"""Implementation of VolatilitySurface object."""

from typing import Optional, Callable

import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance import math
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import implied_volatility_type
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils

_DayCountConventions = daycount_conventions.DayCountConventions
_DayCountConventionsProtoType = types.DayCountConventionsProtoType

interpolation_2d = math.interpolation.interpolation_2d


class VolatilitySurface(pmd.VolatilitySurface):
  """Represents a volatility surface."""

  def __init__(
      self,
      valuation_date: types.DateTensor,
      expiries: types.DateTensor,
      strikes: types.FloatTensor,
      volatilities: types.FloatTensor,
      daycount_convention: Optional[_DayCountConventionsProtoType] = None,
      interpolator: Optional[Callable[[types.FloatTensor, types.FloatTensor],
                                      types.FloatTensor]] = None,
      dtype: Optional[tf.DType] = None,
      name: Optional[str] = None):
    """Initializes the volatility surface.

    Args:
      valuation_date: A `DateTensor` specifying the valuation (or
        settlement) date for the curve.
      expiries: A `DateTensor` containing the expiry dates on which the
        implied volatilities are specified. Should have a compatible shape with
        valuation_date.
      strikes: A `Tensor` of real dtype specifying the strikes corresponding to
        the input maturities. The shape of this input should match the shape of
        `expiries`.
      volatilities: A `Tensor` of real dtype specifying the volatilities
        corresponding to  the input maturities. The shape of this input should
        match the shape of `expiries`.
      daycount_convention: `DayCountConventions` to use for the interpolation
        purpose.
        Default value: `None` which maps to actual/365 day count convention.
      interpolator: An optional Python callable implementing the interpolation
        to be used. The callable should accept two real `Tensor`s specifying
        the strikes and expiry times and should return a real `Tensor` of
        same dtype as the inputs containing the interpolated implied
        volatilities.
        Default value: `None` in which case `Interpolation2D` is used.
      dtype: `tf.Dtype`. Optional input specifying the dtype of the `rates`
        input.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'rate_curve'.
    """
    self._name = name or "VolatilitySurface"
    self._dtype = dtype or tf.float64
    with tf.name_scope(self._name):
      self._daycount_convention = (
          daycount_convention or _DayCountConventions.ACTUAL_365)
      self._day_count_fn = utils.get_daycount_fn(self._daycount_convention)
      self._valuation_date = dateslib.convert_to_date_tensor(
          valuation_date)
      self._expiries = dateslib.convert_to_date_tensor(
          expiries)
      self._strikes = tf.convert_to_tensor(
          strikes, dtype=self._dtype, name="strikes")
      self._volatilities = tf.convert_to_tensor(
          volatilities, dtype=self._dtype, name="volatilities")
      expiry_times = self._day_count_fn(
          start_date=self._valuation_date,
          end_date=self._expiries,
          dtype=self._dtype)
      if interpolator is None:
        interpolator_obj = interpolation_2d.Interpolation2D(
            expiry_times, strikes, volatilities, dtype=self._dtype)
        def interpolator_fn(t, x):
          return interpolator_obj.interpolate(t, x)
        self._interpolator = interpolator_fn
      else:
        self._interpolator = interpolator

  def volatility(self,
                 strike: types.FloatTensor,
                 expiry_dates: Optional[types.DateTensor] = None,
                 expiry_times: Optional[types.FloatTensor] = None,
                 term: Optional[types.Period] = None) -> types.FloatTensor:
    """Returns the interpolated volatility on a specified set of expiries.

    Args:
      strike: The strikes for which the interpolation is desired.
      expiry_dates: Optional input specifying the expiry dates for which
        interpolation is desired. The user should supply either `expiry_dates`
        or `expiry_times` for interpolation.
      expiry_times: Optional real `Tensor` containing the time to expiration
        for which interpolation is desired. The user should supply either
        `expiry_dates` or `expiry_times` for interpolation.
      term: Optional input specifiying the term of the underlying rate for
        which the interpolation is desired. Relevant for interest rate implied
        volatility data.

    Returns:
      A `Tensor` of the same shape as `expiry` with the interpolated volatility
      from the volatility surface.

    Raises:
      ValueError is both `expiry_dates` and `expiry_times`  are specified.
    """
    del term
    if expiry_dates is not None and expiry_times is not None:
      raise ValueError("Unexpected inputs: Both expiry_dates and expiry times "
                       "are specified")

    if expiry_times is None:
      expiry_dates = dateslib.convert_to_date_tensor(expiry_dates)
      expiries = self._day_count_fn(
          start_date=self._valuation_date,
          end_date=expiry_dates,
          dtype=self._dtype)
    else:
      expiries = tf.convert_to_tensor(expiry_times, dtype=self._dtype)
    strike = tf.convert_to_tensor(strike, dtype=self._dtype, name="strike")
    return self._interpolator(expiries, strike)

  def settlement_date(self) -> types.DateTensor:
    """Returns the valuation/settlement date."""
    return self._valuation_date

  def volatility_type(self) -> implied_volatility_type.ImpliedVolatilityType:
    """Returns the type of implied volatility."""
    pass

  def node_expiries(self) -> types.DateTensor:
    """Expiry dates at which the implied volatilities are specified."""
    return self._expiries

  def node_strikes(self) -> tf.Tensor:
    """Strikes at which the implied volatilities are specified."""
    return self._strikes

  def node_volatilities(self) -> tf.Tensor:
    """Market implied volatilities."""
    return self._volatilities

  @property
  def daycount_convention(self) -> _DayCountConventionsProtoType:
    return self._daycount_convention

  def node_terms(self) -> types.Period:
    """Rate terms corresponding to the specified implied volatilities."""
    pass

  def interpolator(self):
    return self._interpolator

__all__ = ["VolatilitySurface"]
