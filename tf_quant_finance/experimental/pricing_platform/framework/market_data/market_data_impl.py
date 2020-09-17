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
"""Implementation of processed market data interface."""

import datetime
from typing import Dict, Any, List, Optional

import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance import math
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import rate_curve
from tf_quant_finance.experimental.pricing_platform.framework.market_data import volatility_surface
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2


class MarketDataDict(pmd.ProcessedMarketData):
  """Market data dictionary representation."""

  def __init__(self,
               valuation_date: types.DateTensor,
               market_data_dict: Dict[str, Any],
               config: Optional[Dict[str, Any]] = None,
               dtype: Optional[tf.DType] = None):
    """Market data constructor.

    ####Example

    ```python
    market_data_dict = {
    "Currency":  {
        "risk_free_curve": {"dates": DateTensor, "discounts": tf.Tensor},
        rate_index : {"dates": DateTensor, "discounts": tf.Tensor},
        surface_id: "to be specified",
        fixings: "to be specified"},
    "Equity": {"spot": FloatTensor,
              "volatility_surface": {"dates": DateTensor,
                                     "strikes": FloatTensor,
                                     "implied_volatilities": FloatTensor}}}
    ```
    Here `curve_index` refers to `curve_types.Index`. The user is expected to
    supply all necessary curves used for pricing. The pricing functions should
    decide which curve to use and how to map `RateIndexType` to
    `curve_types.Index`. Default mapping can be overridden via instrument
    configuration.

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

  def fixings(self,
              date: types.DateTensor,
              fixing_type: curve_types.RateIndexCurve,
              tenor: period_pb2.Period) -> types.FloatTensor:
    """Returns past fixings of the market rates at the specified dates."""
    date = dateslib.convert_to_date_tensor(date)
    return tf.zeros(tf.shape(date.ordinal()),
                    dtype=self._dtype, name="fixings")

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
