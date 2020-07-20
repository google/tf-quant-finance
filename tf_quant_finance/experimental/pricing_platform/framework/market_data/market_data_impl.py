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
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import market_data_config
from tf_quant_finance.experimental.pricing_platform.framework.market_data import rate_curve
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
        curve_index : {"date": DateTensor, "discount": tf.Tensor},
        surface_id(curve_type): "to be specified",
        fixings: "to be specified"},
    "Asset": "to be specified"}
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
    currency = curve_type.currency.value
    if currency not in self.supported_currencies:
      raise ValueError(f"Currency '{curve_type.currency}' is not supported")
    try:
      curve_id = market_data_config.curve_id(curve_type)
      curve_data = self._market_data_dict[currency][curve_id]
    except KeyError:
      raise KeyError(
          "No data for {0} which corresponds to curve {1}".format(
              market_data_config.curve_id(curve_type), curve_type))
    rate_config = None
    if self._config is not None:
      try:
        rate_config = self._config[currency][curve_id]
      except KeyError:
        pass
    dates = curve_data["date"]
    discount_factors = curve_data["discount"]
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
              fixing_type: str,
              tenor: period_pb2.Period) -> types.FloatTensor:
    """Returns past fixings of the market rates at the specified dates."""
    return tf.constant(0, dtype=self._dtype, name="fixings")

  def spot(self, asset: str,
           data: types.DateTensor) -> tf.Tensor:
    """The spot price of an asset."""
    pass

  def volatility_surface(self, asset: str) -> Any:  # To be specified
    """The volatility surface object for an asset."""
    pass

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
    return []

  @property
  def dtype(self) -> types.Dtype:
    """Type of the float calculations."""
    return self._dtype


__all__ = ["MarketDataDict"]
