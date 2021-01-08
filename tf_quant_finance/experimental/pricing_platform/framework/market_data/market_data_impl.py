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
"""Implementation of processed market data interface.

Market data is expected in the following format:

{
  # Currencies
  'USD': {
    'risk_free_curve': {
       # Sorted tensor of discount dates
      'dates': [date1, date2]
      # Discount factors
      'dicounts': [discount1, dicount2]
    }
    'LIBOR_3M': {
       # Sorted tensor of discount dates
      'dates': [date1, date2]
      # Discount factors
      'dicounts': [discount1, dicount2]
      # Sorted tensor of fixing dates
      'fixing_dates': [fixing_date1, fixing_date2, fixing_date3]
      # Fixing rates represented as annualized simple rates
      'fixing_rates': [rate1, rate2, rate3]
      # Fixing daycount. Should have a value that is one of the
      # `core.DayCountConventions`. Must be supplied if the fixings are
      # specified.
      'fixing_daycount': 'ACTUAL_360'
    }
  }
  # Equities
  "GOOG": {
    "spot": 1700,
    "currency": "USD",
    "volatility_surface": {
      "dates": [date1, date2]
      # Strikes for each date. In each row, the last value can be repeated,
      # e.g, [[1650, 1700, 1750, 1800, 1800], [1650, 1700, 1750, 1770, 1790]]
      "strikes": 2d array
      # Implied volatilities corresponding to the dates and strikes.
      "implied_volatilities" 2d array
    }
  }
}

"""

import datetime
from typing import Dict, Any, List, Optional, Tuple

import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance import math
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import rate_curve
from tf_quant_finance.experimental.pricing_platform.framework.market_data import volatility_surface


class MarketDataDict(pmd.ProcessedMarketData):
  """Market data dictionary representation."""

  def __init__(self,
               valuation_date: types.DateTensor,
               market_data_dict: Dict[str, Any],
               config: Optional[Dict[str, Any]] = None,
               dtype: Optional[tf.DType] = None):
    """Market data constructor.

    The dictionary must have the following format

    TODO(b/176962220): Add a doc describign market data structure.
    ```None
    {
      # Currencies
      'USD': {
        'risk_free_curve': {
           # Sorted tensor of discount dates
          'dates': [date1, date2]
          # Discount factors
          'dicounts': [discount1, dicount2]
        }
        'LIBOR_3M': {
           # Sorted tensor of discount dates
          'dates': [date1, date2]
          # Discount factors
          'dicounts': [discount1, dicount2]
          # Sorted tensor of fixing dates
          'fixing_dates': [fixing_date1, fixing_date2, fixing_date3]
          # Fixing rates represented as annualized simple rates
          'fixing_rates': [rate1, rate2, rate3]
          # Fixing daycount. Should have a value that is one of the
          # `core.DayCountConventions`. Must be supplied if the fixings are
          # specified.
          'fixing_daycount': 'ACTUAL_360'
        }
      }
      # Equities
      "GOOG": {
        "spot": 1700,
        "currency": "USD",
        "volatility_surface": {
          "dates": [date1, date2]
        # Strikes for each date. In each row, the last value can be repeated,
        # e.g, [[1650, 1700, 1750, 1800, 1800], [1650, 1700, 1750, 1770, 1790]]
        "strikes": 2d array
        # Implied volatilities corresponding to the dates and strikes.
        "implied_volatilities" 2d array
        }
      }
    }
    ```
    For discounting `risk_free_curve` is used by defualt. Rate indices are
    supplied as strings (e.g., "LIBOR_3M" above) and should have the same name
    as in `rate_indices.RateIndexType`. The user is expected to
    supply all necessary curves used for pricing. The pricing functions should
    decide which curve to use. Configuration argument `config` should be used
    to specify daycount convention and interpolation method used by a curve
    or a volatility surface.

    Args:
      valuation_date: Valuation date.
      market_data_dict: Market data dictionary.
      config: Market data config. See `market_data_config` module description.
        Used to set up rate curve and volatility surface parameters.
      dtype: A `dtype` to use for float-like `Tensor`s.
        Default value: `tf.float64`.
    """
    self._valuation_date = dateslib.convert_to_date_tensor(valuation_date)
    self._market_data_dict = market_data_dict
    self._config = config
    self._dtype = dtype or tf.float64

  @property
  def date(self) -> datetime.date:
    return self._valuation_date

  @property
  def time(self) -> datetime.time:
    """The time of the snapshot."""
    return datetime.time(0)

  def yield_curve(self,
                  curve_type: curve_types.CurveType) -> rate_curve.RateCurve:
    """The yield curve object."""
    # Extract the currency of the curve
    currency = curve_type.currency.value
    if currency not in self.supported_currencies:
      raise ValueError(f"Currency '{curve_type.currency}' is not supported")
    try:
      if isinstance(curve_type, curve_types.RiskFreeCurve):
        curve_id = "risk_free_curve"
      else:
        curve_id = curve_type.index.type.name
      curve_data = self._market_data_dict[currency][curve_id]
    except KeyError:
      raise KeyError(
          "No data for {0} which corresponds to curve {1}".format(
              curve_id, curve_type))
    rate_config = None
    if self._config is not None:
      try:
        rate_config = self._config[currency][curve_id]
      except KeyError:
        pass
    dates = curve_data["dates"]
    discount_factors = curve_data["discounts"]
    if rate_config is None:
      return rate_curve.RateCurve(dates,
                                  discount_factors,
                                  self._valuation_date,
                                  curve_type=curve_type,
                                  dtype=self._dtype)
    else:
      return rate_curve.RateCurve(
          dates,
          discount_factors,
          self._valuation_date,
          curve_type=curve_type,
          interpolator=rate_config.interpolation_method,
          interpolate_rates=rate_config.interpolate_rates,
          daycount_convention=rate_config.daycount_convention,
          dtype=self._dtype)

  def fixings(
      self,
      date: types.DateTensor,
      fixing_type: curve_types.RateIndexCurve
      ) -> Tuple[tf.Tensor, daycount_conventions.DayCountConventions]:
    """Returns past fixings of the market rates at the specified dates.

    The fixings are represented asannualized simple rates. When fixings are not
    provided for a curve, they are assumed to be zero for any date. Otherwise,
    it is assumed that the fixings are a left-continuous piecewise-constant
    of time with jumps being the supplied fixings.

    Args:
      date: The dates at which the fixings are computed. Should precede the
        valuation date. When passed as an integet `Tensor`, should be of shape
        `batch_shape + [3]` and contain `[year, month, day]` for each date.
      fixing_type: Rate index curve type for which the fixings are computed.

    Returns:
      A `Tensor` of the same shape of `date` and of `self.dtype` dtype.
      Represents fixings at the requested `date`.
    """
    index_type = fixing_type.index.type.value
    currency = fixing_type.currency.value
    if isinstance(date, tf.Tensor):
      # When the input is a Tensor, `dateslib.convert_to_date_tensor` assumes
      # that the ordinals are passed. Instead we assume that the inputs
      # are of shape `batch_shape + [3]` and are interpreted as pairs
      # [year, month, day]
      date = dateslib.dates_from_tensor(date)
    else:
      date = dateslib.convert_to_date_tensor(date)
    try:
      curve_data = self._market_data_dict[currency][index_type]
      fixing_dates = curve_data["fixing_dates"]
      fixing_rates = curve_data["fixing_rates"]
    except KeyError:
      return tf.zeros(tf.shape(date.ordinal()),
                      dtype=self._dtype, name="fixings"), None
    if isinstance(fixing_dates, tf.Tensor):
      fixing_dates = dateslib.dates_from_tensor(fixing_dates)
    else:
      fixing_dates = dateslib.convert_to_date_tensor(fixing_dates)
    if "fixing_daycount" not in curve_data:
      raise ValueError(
          f"`fixing_daycount` should be specified for {index_type}.")
    fixing_daycount = curve_data["fixing_daycount"]
    fixing_daycount = daycount_conventions.DayCountConventions(fixing_daycount)
    fixing_rates = tf.convert_to_tensor(fixing_rates, dtype=self._dtype)
    fixing_dates_ordinal = fixing_dates.ordinal()
    date_ordinal = date.ordinal()
    # Broadcast fixing dates for tf.searchsorted
    batch_shape = tf.shape(date_ordinal)[:-1]
    # Broadcast valuation date batch shape for tf.searchsorted
    fixing_dates_ordinal += tf.expand_dims(
        tf.zeros(batch_shape, dtype=tf.int32), axis=-1)
    inds = tf.searchsorted(fixing_dates_ordinal, date_ordinal)
    inds = tf.maximum(inds, 0)
    inds = tf.minimum(inds, tf.shape(fixing_dates_ordinal)[-1] - 1)
    return tf.gather(fixing_rates, inds), fixing_daycount

  def spot(self, asset: List[str],
           date: types.DateTensor = None) -> tf.Tensor:
    """The spot price of an asset."""
    spots = []
    for s in asset:
      if s not in self.supported_assets:
        raise ValueError(f"No data for asset {s}")
      data_s = self._market_data_dict[s]
      spots.append(tf.convert_to_tensor(data_s["spot"], self._dtype))
    return spots

  def volatility_surface(
      self, asset: List[str]) -> volatility_surface.VolatilitySurface:
    """The volatility surface object for the lsit of assets.

    Args:
      asset: A list of strings with asset names.

    Returns:
      An instance of `VolatilitySurface`.
    """
    dates = []
    strikes = []
    implied_vols = []
    for s in asset:
      if s not in self.supported_assets:
        raise ValueError(f"No data for asset {s}")
      data_s = self._market_data_dict[s]
      if "volatility_surface" not in data_s:
        raise ValueError(
            f"No volatility surface 'volatility_surface' for asset {s}")
      vol_surface = data_s["volatility_surface"]
      vol_dates = dateslib.convert_to_date_tensor(vol_surface["dates"])
      vol_strikes = tf.convert_to_tensor(
          vol_surface["strikes"], dtype=self._dtype, name="strikes")
      vols = tf.convert_to_tensor(
          vol_surface["implied_volatilities"], dtype=self._dtype,
          name="implied_volatilities")
      dates.append(vol_dates)
      strikes.append(vol_strikes)
      implied_vols.append(vols)
    dates = math.pad.pad_date_tensors(dates)
    dates = dateslib.DateTensor.stack(dates, axis=0)
    implied_vols = math.pad.pad_tensors(implied_vols)
    implied_vols = tf.stack(implied_vols, axis=0)
    strikes = math.pad.pad_tensors(strikes)
    strikes = tf.stack(strikes, axis=0)
    vol_surface = volatility_surface.VolatilitySurface(
        self.date, dates, strikes, implied_vols)
    return vol_surface

  def forward_curve(self, asset: str):
    """The forward curve of the asset prices object."""
    pass

  @property
  def supported_currencies(self) -> List[str]:
    """List of supported currencies."""
    return [kk for kk in self._market_data_dict.keys()
            if kk in currencies.Currency.__dict__["_member_names_"]]

  @property
  def supported_assets(self) -> List[str]:
    """List of supported assets."""
    market_keys = self._market_data_dict.keys()
    return [k for k in market_keys if k not in self.supported_currencies]

  @property
  def dtype(self) -> types.Dtype:
    """Type of the float calculations."""
    return self._dtype


__all__ = ["MarketDataDict"]
