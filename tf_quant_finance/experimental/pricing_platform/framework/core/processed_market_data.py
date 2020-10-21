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
"""Interface for the Market data."""

import abc
import datetime
from typing import Any, List, Tuple, Callable, Optional

import tensorflow.compat.v2 as tf

from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types

from tf_quant_finance.experimental.pricing_platform.framework.core import implied_volatility_type
from tf_quant_finance.experimental.pricing_platform.framework.core import interpolation_method
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2


class RateCurve(abc.ABC):
  """Interface for interest rate curves."""

  @abc.abstractmethod
  def discount_factor(self,
                      date: Optional[types.DateTensor] = None,
                      time: Optional[types.FloatTensor] = None,
                      **kwargs) -> tf.Tensor:
    """Returns the discount factor to a specified set of dates.

    Args:
      date: Optional input specifying the dates at which to evaluate the
        discount factors. The function expects either `date` or `time` to be
        specified.
      time: Optional input specifying the times at which to evaluate the
        discount factors. The function expects either `date` or `time` to be
        specified.
      **kwargs: The context object, e.g., curve_type.

    Returns:
      A `Tensor` of the same shape as `dates` with the corresponding discount
      factors.
    """
    pass

  @abc.abstractmethod
  def forward_rate(self,
                   start_date: Optional[types.DateTensor] = None,
                   end_date: Optional[types.DateTensor] = None,
                   start_time: Optional[types.FloatTensor] = None,
                   end_time: Optional[types.FloatTensor] = None,
                   **kwargs) -> tf.Tensor:
    """Returns the simply accrued forward rate between dates.

    Args:
      start_date: A `DateTensor` specifying the start of the accrual period
        for the forward rate. The function expects either `start_date` or
        `start_time` to be specified.
      end_date: A `DateTensor` specifying the end of the accrual period
        for the forward rate. The shape of `end_date` must be broadcastable
        with the shape of `start_date`. The function expects either `end_date`
        or `end_time` to be specified.
      start_time: A real `Tensor` specifying the start of the accrual period
        for the forward rate. The function expects either `start_date` or
        `start_time` to be specified.
      end_time: A real `Tensor` specifying the end of the accrual period
        for the forward rate. The shape of `end_date` must be broadcastable
        with the shape of `start_date`. The function expects either `end_date`
        or `end_time` to be specified.
      **kwargs: The context object, e.g., curve_type.

    Returns:
      A `Tensor` with the corresponding forward rates.
    """
    pass

  @abc.abstractmethod
  def discount_rate(self,
                    date: Optional[types.DateTensor] = None,
                    time: Optional[types.FloatTensor] = None,
                    context=None) -> tf.Tensor:
    """Returns the discount rates to a specified set of dates.

    Args:
      date: A `DateTensor` specifying the dates at which to evaluate the
        discount rates. The function expects either `date` or `time` to be
        specified.
      time: A real `Tensor` specifying the times at which to evaluate the
        discount rates. The function expects either `date` or `time` to be
        specified.
      context: The context object, e.g., curve_type.

    Returns:
      A `Tensor` of the same shape as `dates` with the corresponding discount
      rates.
    """
    pass

  @property
  @abc.abstractmethod
  def curve_type(self) ->Any:  # to be specified
    """Returns type of the curve."""
    pass

  @abc.abstractmethod
  def interpolation_method(self) -> interpolation_method.InterpolationMethod:
    """Interpolation method used for this discount curve."""
    pass

  @abc.abstractmethod
  def discount_factors_and_dates(self) -> Tuple[types.FloatTensor,
                                                types.DateTensor]:
    """Returns discount factors and dates at which the discount curve is fitted.
    """
    pass

  @abc.abstractproperty
  def discount_factor_nodes(self) -> types.FloatTensor:
    """Discount factors at the interpolation nodes."""
    pass

  @abc.abstractmethod
  def set_discount_factor_nodes(self,
                                values: types.FloatTensor) -> types.FloatTensor:
    """Update discount factors at the interpolation nodes with new values."""
    pass

  @abc.abstractproperty
  def discount_rate_nodes(self) -> types.FloatTensor:
    """Discount rates at the interpolation nodes."""
    pass

  @abc.abstractproperty
  def node_dates(self) -> types.DateTensor:
    """Dates at which the discount factors and rates are specified."""
    return self._dates

  @abc.abstractproperty
  def daycount_convention(self) -> types.DayCountConventionsProtoType:
    """Daycount convention."""
    raise NotImplementedError

  @abc.abstractmethod
  def daycount_fn(self) -> Callable[..., Any]:
    """Daycount function."""
    raise NotImplementedError


class VolatilitySurface(abc.ABC):
  """Interface for implied volatility surface."""

  @abc.abstractmethod
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
    """
    pass

  @property
  @abc.abstractmethod
  def volatility_type(self) -> implied_volatility_type.ImpliedVolatilityType:
    """Returns the type of implied volatility."""
    pass

  @property
  @abc.abstractmethod
  def node_expiries(self) -> types.DateTensor:
    """Expiry dates at which the implied volatilities are specified."""
    return self._expiries

  @property
  @abc.abstractmethod
  def node_strikes(self) -> tf.Tensor:
    """Striks at which the implied volatilities are specified."""
    return self._strikes

  @property
  @abc.abstractmethod
  def node_terms(self) -> types.Period:
    """Rate terms corresponding to the specified implied volatilities."""
    return self._terms


class ProcessedMarketData(abc.ABC):
  """Market data snapshot used by pricing library."""

  @abc.abstractproperty
  def date(self) -> datetime.date:
    """The date of the market data."""
    pass

  @abc.abstractproperty
  def time(self) -> datetime.time:
    """The time of the snapshot."""
    pass

  @abc.abstractmethod
  def yield_curve(self, curve_type: curve_types.CurveType) -> RateCurve:
    """The yield curve object."""
    pass

  @abc.abstractmethod
  def fixings(self,
              date: types.DateTensor,
              fixing_type: curve_types.RateIndexCurve,
              tenor: period_pb2.Period) -> tf.Tensor:
    """Returns past fixings of the market rates at the specified dates."""
    pass

  @abc.abstractmethod
  def spot(self, asset: str,
           data: types.DateTensor) -> tf.Tensor:
    """The spot price of an asset."""
    pass

  @abc.abstractmethod
  def volatility_surface(self, asset: str) -> VolatilitySurface:
    """The volatility surface object for an asset."""
    pass

  @abc.abstractmethod
  def forward_curve(self, asset: str) -> RateCurve:
    """The forward curve of the asset prices object."""
    pass

  @abc.abstractproperty
  def supported_currencies(self) -> List[str]:
    """List of supported currencies."""
    pass

  @abc.abstractmethod
  def supported_assets(self) -> List[str]:
    """List of supported assets."""
    pass

  @abc.abstractproperty
  def dtype(self) -> types.Dtype:
    """Type of the float calculations."""
    pass


__all__ = ["RateCurve", "ProcessedMarketData"]

