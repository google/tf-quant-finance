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
"""Equity american option."""

from typing import Any, Optional, List, Dict, Union, Tuple

import dataclasses
import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types as curve_types_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import instrument
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.equity_instruments import utils as equity_utils
from tf_quant_finance.experimental.pricing_platform.framework.equity_instruments.american_option import proto_utils
from tf_quant_finance.experimental.pricing_platform.framework.equity_instruments.american_option import utils
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils as market_data_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import cashflow_streams
from tf_quant_finance.experimental.pricing_platform.instrument_protos import american_equity_option_pb2 as american_option_pb2


@dataclasses.dataclass(frozen=True)
class AmericanOptionConfig:
  discounting_curve: Optional[
      Dict[types.CurrencyProtoType,
           curve_types_lib.CurveType]] = dataclasses.field(default_factory=dict)
  model: str = "BS-LSM"  # default pricing model is LSM under Black-Scholes
  num_samples: int = 96000
  num_calibration_samples: int = None
  num_exercise_times: int = 100
  seed: types.IntTensor = (42, 42)  # Should be an integer `Tensor` of shape [2]


class AmericanOption(instrument.Instrument):
  """Represents a batch of American Equity Options.

  An American equity option is a contract that gives the holder an opportunity
  to buy (call) or sell (put) an equity for a predefined value (strike) at
  any date before the expiry.

  The AmericanOption class can be used to create and price multiple options
  simultaneously.

  #### Example:
  The following example illustrates the construction of a batch of American
  options and pricing them.

  ```None
  american_option_proto = american_option_pb2.AmericanEquityOption(
      short_position=True,
      expiry_date=date_pb2.Date(year=2022, month=5, day=21),
      contract_amount=decimal_pb2.Decimal(units=10000),
      strike=decimal_pb2.Decimal(units=1500),
      equity="GOOG",
      currency=Currency.USD(),
      business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
      is_call_option=False)

  market = market_data.MarketDataDict(...)
  am_option_portfolio = AmericanOption.from_protos([american_option_proto])
  am_option_portfolio[0].price(market)
  ```
  """

  def __init__(self,
               short_position: types.BoolTensor,
               currency: Union[types.CurrencyProtoType,
                               List[types.CurrencyProtoType]],
               expiry_date: types.DateTensor,
               equity: List[str],
               contract_amount: types.FloatTensor,
               strike: types.FloatTensor,
               is_call_option: List[bool],
               business_day_convention: types.BusinessDayConventionProtoType,
               calendar: types.BankHolidaysProtoType,
               settlement_days: Optional[types.IntTensor] = 0,
               discount_curve_type: curve_types_lib.CurveType = None,
               discount_curve_mask: types.IntTensor = None,
               equity_mask: types.IntTensor = None,
               config: Union[AmericanOptionConfig, Dict[str, Any]] = None,
               batch_names: Optional[types.StringTensor] = None,
               dtype: Optional[types.Dtype] = None,
               name: Optional[str] = None):
    """Initializes the batch of American Equity Options.

    Args:
      short_position: Whether the price is computed for the contract holder.
        Default value: `True` which means that the price is for the contract
        holder.
      currency: The denominated currency.
      expiry_date: A `DateTensor` specifying the dates on which the options
        expire.
      equity: A string name of the underlyings.
      contract_amount: A `Tensor` of real dtype and shape compatible with
        with `short_position`.
      strike: `Tensor` of real dtype and shape compatible with
        with `short_position`. Option strikes.
      is_call_option: A bool `Tensor` of shape compatible with with
        `short_position`. Indicates which options are of call type.
      business_day_convention: A business count convention.
      calendar: A calendar to specify the weekend mask and bank holidays.
      settlement_days: An integer `Tensor` of the shape broadcastable with the
        shape of `fixing_date`.
      discount_curve_type: An optional instance of `CurveType` or a list of
        those. If supplied as a list and `discount_curve_mask` is not supplied,
        the size of the list should be the same as the number of priced
        instruments. Defines discount curves for the instruments.
        Default value: `None`, meaning that discount curves are inferred
        from `currency` and `config`.
      discount_curve_mask: An optional integer `Tensor` of values ranging from
        `0` to `len(discount_curve_type) - 1` and of shape `batch_shape`.
        Identifies a mapping between `discount_curve_type` list and the
        underlying instruments.
        Default value: `None`.
      equity_mask: An optional integer `Tensor` of values ranging from
        `0` to `len(equity) - 1` and of shape `batch_shape`. Identifies
        a mapping between `equity` list and the underlying instruments.
        Default value: `None`.
      config: Optional `AmericanOptionConfig` or a dictionary. If dictionary,
        then the keys should be the same as the field names of
        `AmericanOptionConfig`.
      batch_names: A string `Tensor` of instrument names. Should be of shape
        `batch_shape + [2]` specying name and instrument type. This is useful
        when the `from_protos` method is used and the user needs to identify
        which instruments got batched together.
      dtype: `tf.Dtype` of the input and output real `Tensor`s.
        Default value: `None` which maps to `float64`.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'AmericanOption'.
    """
    self._name = name or "AmericanOption"
    with tf.name_scope(self._name):
      if batch_names is not None:
        self._names = tf.convert_to_tensor(batch_names,
                                           name="batch_names")
      else:
        self._names = None
      self._dtype = dtype or tf.float64
      ones = tf.constant(1, dtype=self._dtype)
      self._short_position = tf.where(
          short_position, ones, -ones, name="short_position")
      self._contract_amount = tf.convert_to_tensor(
          contract_amount, dtype=self._dtype, name="contract_amount")
      self._strike = tf.convert_to_tensor(strike, dtype=self._dtype,
                                          name="strike")
      self._is_call_option = tf.convert_to_tensor(
          is_call_option, dtype=tf.bool, name="strike")
      settlement_days = tf.convert_to_tensor(settlement_days)
      # Business day roll convention and the end of month flag
      roll_convention, eom = market_data_utils.get_business_day_convention(
          business_day_convention)
      # TODO(b/160446193): Calendar is ignored at the moment
      calendar = dateslib.create_holiday_calendar(
          weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
      if isinstance(expiry_date, types.IntTensor):
        self._expiry_date = dateslib.dates_from_tensor(expiry_date)
      else:
        self._expiry_date = dateslib.convert_to_date_tensor(expiry_date)
      self._settlement_days = settlement_days
      self._roll_convention = roll_convention
      # Get discount and reference curves
      self._currency = cashflow_streams.to_list(currency)
      self._equity = cashflow_streams.to_list(equity)
      if len(self._currency) != len(self._equity):
        if len(self._currency) > 1 and len(self._equity) > 1:
          raise ValueError(
              "Number of currencies and equities should be the same "
              "but it is {0} and {1}".format(len(self._currency),
                                             len(self._equity)))

      config = _process_config(config)
      [
          self._model,
          self._num_samples,
          self._seed,
          self._num_exercise_times,
          self._num_calibration_samples
      ] = _get_config_values(config)

      if discount_curve_type is None:
        discount_curve_type = []
        for currency in self._currency:
          if currency in config.discounting_curve:
            curve_type = config.discounting_curve[currency]
          else:
            # Default discounting curve
            curve_type = curve_types_lib.RiskFreeCurve(
                currency=currency)
          discount_curve_type.append(curve_type)

      # Get masks for discount curves and vol surfaces
      [
          self._discount_curve_type,
          self._discount_curve_mask
      ] = cashflow_streams.process_curve_types(discount_curve_type,
                                               discount_curve_mask)
      [
          self._equity,
          self._equity_mask,
      ] = equity_utils.process_equities(self._equity, equity_mask)
      # Get batch shape
      self._batch_shape = tf.shape(strike)

  @classmethod
  def create_constructor_args(
      cls, proto_list: List[american_option_pb2.AmericanEquityOption],
      config: AmericanOptionConfig = None) -> Dict[str, Any]:
    """Creates a dictionary to initialize AmericanEquityOption.

    The output dictionary is such that the instruments can be initialized
    as follows:
    ```
    initializer = create_constructor_args(proto_list, config)
    american_options = [AmericanEquityOption(**data)
                        for data in initializer.values()]
    ```

    The keys of the output dictionary are unique identifiers of the batched
    instruments. This is useful for identifying an existing graph that could be
    reused for the instruments without the need of rebuilding the graph.

    Args:
      proto_list: A list of protos for which the initialization arguments are
        constructed.
      config: An instance of `AmericanOptionConfig`.

    Returns:
      A possibly nested dictionary such that each value provides initialization
      arguments for the AmericanEquityOption.
    """
    am_option_data = proto_utils.from_protos(proto_list, config)
    res = {}
    for key in am_option_data:
      tensor_repr = proto_utils.tensor_repr(am_option_data[key])
      res[key] = tensor_repr
    return res

  @classmethod
  def from_protos(
      cls,
      proto_list: List[american_option_pb2.AmericanEquityOption],
      config: AmericanOptionConfig = None
      ) -> List["AmericanOption"]:
    proto_dict = proto_utils.from_protos(proto_list, config)
    instruments = []
    for kwargs in proto_dict.values():
      # Create an instrument
      instruments.append(cls(**kwargs))
    return instruments

  @classmethod
  def group_protos(
      cls,
      proto_list: List[american_option_pb2.AmericanEquityOption],
      config: AmericanOptionConfig = None
      ) -> Dict[str, List["AmericanOption"]]:
    return proto_utils.group_protos(proto_list, config)

  def price(self,
            market: pmd.ProcessedMarketData,
            name: Optional[str] = None) -> types.FloatTensor:
    """Returns the present value of the American options.

    Args:
      market: An instance of `ProcessedMarketData`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A `Tensor` of shape `batch_shape`  containing the modeled price of each
      American option contract based on the input market data.
    """
    name = name or (self._name + "_price")
    with tf.name_scope(name):
      discount_curve = cashflow_streams.get_discount_curve(
          self._discount_curve_type, market, self._discount_curve_mask)
      currencies = [cur.currency.value for cur in self._discount_curve_type]
      vol_surface = equity_utils.get_vol_surface(
          currencies, self._equity, market, self._equity_mask)
      spots = tf.stack(market.spot(currencies, self._equity), axis=0)
      discount_factors = discount_curve.discount_factor(
          self._expiry_date.expand_dims(axis=-1))
      daycount_convention = discount_curve.daycount_convention
      day_count_fn = market_data_utils.get_daycount_fn(daycount_convention)
      if spots.shape.rank > 0:
        spots = tf.gather(spots, self._equity_mask)
      if self._model == "BS-LSM":
        # TODO(b/168798725): volatility should be time-dependent
        vols = vol_surface.volatility(
            expiry_dates=self._expiry_date.expand_dims(axis=-1),
            strike=tf.expand_dims(self._strike, axis=-1))
        prices = utils.bs_lsm_price(
            spots=spots,
            expiry_times=day_count_fn(
                start_date=market.date,
                end_date=self._expiry_date,
                dtype=self._dtype),
            strikes=self._strike,
            volatility=tf.squeeze(vols, axis=-1),
            discount_factors=tf.squeeze(discount_factors),
            is_call_option=self._is_call_option,
            num_samples=self._num_samples,
            num_exercise_times=self._num_exercise_times,
            num_calibration_samples=self._num_calibration_samples,
            seed=self._seed)
        return self._short_position * self._contract_amount * prices
      else:
        raise ValueError("Only BS-LSM model is supported. "
                         "Supplied {}".format(self._model))

  @property
  def batch_shape(self) -> tf.Tensor:
    return self._batch_shape

  @property
  def names(self) -> tf.Tensor:
    """Returns a string tensor of names and instrument types.

    The shape of the output is  [batch_shape, 2].
    """
    return self._names

  def ir_delta(self,
               tenor: types.DateTensor,
               processed_market_data: pmd.ProcessedMarketData,
               curve_type: Optional[curve_types_lib.CurveType] = None,
               shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes delta wrt to the tenor perturbation."""
    raise NotImplementedError("Coming soon.")

  def ir_delta_parallel(
      self,
      processed_market_data: pmd.ProcessedMarketData,
      curve_type: Optional[curve_types_lib.CurveType] = None,
      shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes delta wrt to the curve parallel perturbation."""
    raise NotImplementedError("Coming soon.")

  def ir_vega(self,
              tenor: types.DateTensor,
              processed_market_data: pmd.ProcessedMarketData,
              shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes vega wrt to the tenor perturbation."""
    raise NotImplementedError("Coming soon.")


def _process_config(
    config: Union[AmericanOptionConfig, Dict[str, Any], None]
    ) -> AmericanOptionConfig:
  """Converts config to AmericanOptionConfig."""
  if config is None:
    return AmericanOptionConfig()
  if isinstance(config, AmericanOptionConfig):
    return config
  model = config.get("model", "BS-LSM")
  seed = config.get("seed", [42, 42])
  num_exercise_times = config.get("num_exercise_times", 100)
  num_samples = config.get("num_samples", 96000)
  num_calibration_samples = config.get("num_calibration_samples", None)
  discounting_curve = config.get("discounting_curve", dict())
  return AmericanOptionConfig(discounting_curve=discounting_curve,
                              model=model,
                              seed=seed,
                              num_exercise_times=num_exercise_times,
                              num_samples=num_samples,
                              num_calibration_samples=num_calibration_samples)


def _get_config_values(
    config: AmericanOptionConfig
    ) -> Tuple[str, int, types.IntTensor, int, int]:
  """Extracts config values."""
  [
      model,
      num_samples,
      seed,
      num_exercise_times,
      num_calibration_samples
  ] = [config.model,
       config.num_samples,
       tf.convert_to_tensor(config.seed, name="seed"),
       config.num_exercise_times,
       config.num_calibration_samples]
  return model, num_samples, seed, num_exercise_times, num_calibration_samples


__all__ = ["AmericanOptionConfig", "AmericanOption"]
