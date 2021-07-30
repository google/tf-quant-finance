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
"""Forward Rate Agreement."""

from typing import Any, Optional, List, Dict, Union

import dataclasses
import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types as curve_types_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import instrument
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import rate_indices
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils as market_data_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import cashflow_streams
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.forward_rate_agreement import proto_utils
from tf_quant_finance.experimental.pricing_platform.instrument_protos import forward_rate_agreement_pb2 as fra
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2


@dataclasses.dataclass(frozen=True)
class ForwardRateAgreementConfig:
  discounting_curve: Dict[
      types.CurrencyProtoType,
      curve_types_lib.CurveType] = dataclasses.field(default_factory=dict)
  model: str = ""


class ForwardRateAgreement(instrument.Instrument):
  """Represents a batch of Forward Rate Agreements (FRA).

  An FRA is a contract for the period [T, T+tau] where the holder exchanges a
  fixed rate (agreed at the start of the contract) against a floating payment
  determined at time T based on the spot Libor rate for term `tau`. The
  cashflows are exchanged at the settlement time T_s, which is either equal to T
  or close to T. See, e.g., [1].

  The ForwardRateAgreement class can be used to create and price multiple FRAs
  simultaneously. However all FRAs within an FRA object must be priced using
  a common reference and discount curve.

  #### Example:
  The following example illustrates the construction of an FRA instrument and
  calculating its price.

  ```python
  RateIndex = instrument_protos.rate_indices.RateIndex

  fra = fra_pb2.ForwardRateAgreement(
      short_position=True,
      fixing_date=date_pb2.Date(year=2021, month=5, day=21),
      currency=Currency.USD(),
      fixed_rate=decimal_pb2.Decimal(nanos=31340000),
      notional_amount=decimal_pb2.Decimal(units=10000),
      daycount_convention=DayCountConventions.ACTUAL_360(),
      business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
      floating_rate_term=fra_pb2.FloatingRateTerm(
          floating_rate_type=RateIndex(type="LIBOR_3M"),
          term = period_pb2.Period(type="MONTH", amount=3)),
      settlement_days=2)
  date = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
          [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
  discount = [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
              0.72494879, 0.37602059]
  market_data_dict = {
      "rates": {
          "USD": {
              "risk_free_curve": {
                  "dates": dates,
                  "discounts": discounts,
              },
              "LIBOR_3M": {
                  "dates": dates,
                  "discounts": discounts,
              }
          }
      },
      "reference_date": [(2020, 2, 8)],
  }
  market = market_data.MarketDataDict(market_data_dict)
  fra_portfolio = forward_rate_agreement.ForwardRateAgreement.from_protos([fra])
  fra_portfolio[0].price(market)
  # Expected result: [4.05463257]
  ```

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

  def __init__(self,
               short_position: types.BoolTensor,
               currency: types.CurrencyProtoType,
               fixing_date: types.DateTensor,
               fixed_rate: types.FloatTensor,
               notional_amount: types.FloatTensor,
               daycount_convention: types.DayCountConventionsProtoType,
               business_day_convention: types.BusinessDayConventionProtoType,
               calendar: types.BankHolidaysProtoType,
               rate_term: period_pb2.Period,
               rate_index: rate_indices.RateIndex,
               settlement_days: Optional[types.IntTensor] = 0,
               discount_curve_type: curve_types_lib.CurveType = None,
               discount_curve_mask: types.IntTensor = None,
               rate_index_curves: curve_types_lib.RateIndexCurve = None,
               reference_mask: types.IntTensor = None,
               config: Union[ForwardRateAgreementConfig, Dict[str, Any]] = None,
               batch_names: Optional[types.StringTensor] = None,
               dtype: Optional[types.Dtype] = None,
               name: Optional[str] = None):
    """Initializes the batch of FRA contracts.

    Args:
      short_position: Whether the contract holder lends or borrows the money.
        Default value: `True` which means that the contract holder lends the
        money at the fixed rate.
      currency: The denominated currency.
      fixing_date: A `DateTensor` specifying the dates on which forward
        rate will be fixed.
      fixed_rate: A `Tensor` of real dtype specifying the fixed rate
        payment agreed at the initiation of the individual contracts. The shape
        should be broadcastable with `fixed_rate`.
      notional_amount: A `Tensor` of real dtype broadcastable with fixed_rate
        specifying the notional amount for each contract. When the notional is
        specified as a scalar, it is assumed that all contracts have the same
        notional.
      daycount_convention: A `DayCountConvention` to determine how cashflows
        are accrued for each contract. Daycount is assumed to be the same for
        all contracts in a given batch.
      business_day_convention: A business count convention.
      calendar: A calendar to specify the weekend mask and bank holidays.
      rate_term: A tenor of the rate (usually Libor) that determines the
        floating cashflow.
      rate_index: A type of the floating leg. An instance of
        `core.rate_indices.RateIndex`.
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
      rate_index_curves: An instance of `RateIndexCurve` or a list of those.
        If supplied as a list and `reference_mask` is not supplid,
        the size of the list should be the same as the number of priced
        instruments. Defines the index curves for each instrument. If not
        supplied, `coupon_spec.floating_rate_type` is used to identify the
        curves.
        Default value: `None`.
      reference_mask: An optional integer `Tensor` of values ranging from
        `0` to `len(rate_index_curves) - 1` and of shape `batch_shape`.
        Identifies a mapping between `rate_index_curves` list and the underlying
        instruments.
        Default value: `None`.
      config: Optional `ForwardRateAgreementConfig` or a dictionary.
        If dictionary, then the keys should be the same as the field names of
        `ForwardRateAgreementConfig`.
      batch_names: A string `Tensor` of instrument names. Should be of shape
        `batch_shape + [2]` specying name and instrument type. This is useful
        when the `from_protos` method is used and the user needs to identify
        which instruments got batched together.
      dtype: `tf.Dtype` of the input and output real `Tensor`s.
        Default value: `None` which maps to `float64`.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'forward_rate_agreement'.
    """
    self._name = name or "forward_rate_agreement"
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
      self._notional_amount = tf.convert_to_tensor(
          notional_amount, dtype=self._dtype, name="notional_amount")
      self._fixed_rate = tf.convert_to_tensor(fixed_rate, dtype=self._dtype,
                                              name="fixed_rate")
      settlement_days = tf.convert_to_tensor(settlement_days)
      # Business day roll convention and the end of month flag
      roll_convention, eom = market_data_utils.get_business_day_convention(
          business_day_convention)
      # TODO(b/160446193): Calendar is ignored at the moment
      calendar = dateslib.create_holiday_calendar(
          weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
      if isinstance(fixing_date, types.IntTensor):
        self._fixing_date = dateslib.dates_from_tensor(fixing_date)
      else:
        self._fixing_date = dateslib.convert_to_date_tensor(fixing_date)
      self._accrual_start_date = calendar.add_business_days(
          self._fixing_date, settlement_days, roll_convention=roll_convention)

      self._day_count_fn = market_data_utils.get_daycount_fn(
          daycount_convention)
      period = rate_term
      if isinstance(rate_term, period_pb2.Period):
        period = market_data_utils.get_period(rate_term)
      if isinstance(rate_term, dict):
        period = market_data_utils.period_from_dict(rate_term)
      self._accrual_end_date = calendar.add_period_and_roll(
          self._accrual_start_date, period,
          roll_convention=roll_convention)
      if eom:
        self._accrual_end_date = self._accrual_end_date.to_end_of_month()
      self._daycount_fractions = self._day_count_fn(
          start_date=self._accrual_start_date,
          end_date=self._accrual_end_date,
          dtype=self._dtype)
      self._settlement_days = settlement_days
      self._roll_convention = roll_convention
      # Get discount and reference curves
      self._currency = cashflow_streams.to_list(currency)
      self._rate_index = cashflow_streams.to_list(rate_index)
      # Get a mask for the reference curves
      if rate_index_curves is None:
        rate_index_curves = []
        if len(self._currency) != len(self._rate_index):
          raise ValueError(
              "When rate_index_curves` is not supplied, number of currencies "
              "and rate indices should be the same `but it is {0} and "
              "{1}".format(len(self._currency), len(self._rate_index)))

        for currency, rate_index in zip(self._currency,
                                        self._rate_index):
          rate_index_curves.append(curve_types_lib.RateIndexCurve(
              currency=currency, index=rate_index))
      [
          self._reference_curve_type,
          self._reference_mask
      ] = cashflow_streams.process_curve_types(rate_index_curves,
                                               reference_mask)
      # Get a mask for the discount curves
      self._config = _process_config(config)
      if discount_curve_type is None:
        curve_list = []
        for currency in self._currency:
          if currency in self._config.discounting_curve:
            discount_curve_type = self._config.discounting_curve[currency]
          else:
            # Default discounting is the risk free curve
            discount_curve_type = curve_types_lib.RiskFreeCurve(
                currency=currency)
          curve_list.append(discount_curve_type)
      else:
        curve_list = cashflow_streams.to_list(discount_curve_type)

      # Get masks for discount and reference curves
      [
          self._discount_curve_type,
          self._mask
      ] = cashflow_streams.process_curve_types(curve_list, discount_curve_mask)

      # Get batch shape
      self._batch_shape = self._daycount_fractions.shape.as_list()[:-1]

  @classmethod
  def create_constructor_args(
      cls, proto_list: List[fra.ForwardRateAgreement],
      config: ForwardRateAgreementConfig = None) -> Dict[str, Any]:
    """Creates a dictionary to initialize ForwardRateAgreement.

    The output dictionary is such that the instruments can be initialized
    as follows:
    ```
    initializer = create_constructor_args(proto_list, config)
    fras = [ForwardRateAgreement(**data) for data in initializer.values()]
    ```

    The keys of the output dictionary are unique identifiers of the batched
    instruments. This is useful for identifying an existing graph that could be
    reused for the instruments without the need of rebuilding the graph.

    Args:
      proto_list: A list of protos for which the initialization arguments are
        constructed.
      config: An instance of `ForwardRateAgreementConfig`.

    Returns:
      A possibly nested dictionary such that each value provides initialization
      arguments for the ForwardRateAgreement.
    """
    fra_data = proto_utils.from_protos_v2(proto_list, config)
    res = {}
    for key in fra_data:
      tensor_repr = proto_utils.tensor_repr(fra_data[key])
      res[key] = tensor_repr
    return res

  @classmethod
  def from_protos(
      cls,
      proto_list: List[fra.ForwardRateAgreement],
      config: ForwardRateAgreementConfig = None
      ) -> List["ForwardRateAgreement"]:
    proto_dict = proto_utils.from_protos_v2(proto_list, config)
    instruments = []
    for kwargs in proto_dict.values():
      # Convert rate term to the period tensors
      kwargs["rate_term"] = market_data_utils.period_from_list(
          kwargs["rate_term"])
      # Create an instrument
      instruments.append(cls(**kwargs))
    return instruments

  @classmethod
  def group_protos(
      cls,
      proto_list: List[fra.ForwardRateAgreement],
      config: ForwardRateAgreementConfig = None
      ) -> Dict[str, List["ForwardRateAgreement"]]:
    return proto_utils.group_protos_v2(proto_list, config)

  def price(self,
            market: pmd.ProcessedMarketData,
            name: Optional[str] = None) -> types.FloatTensor:
    """Returns the present value of the stream on the valuation date.

    Args:
      market: An instance of `ProcessedMarketData`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A `Tensor` of shape `batch_shape`  containing the modeled price of each
      FRA contract based on the input market data.
    """
    name = name or (self._name + "_price")
    with tf.name_scope(name):
      discount_curve = cashflow_streams.get_discount_curve(
          self._discount_curve_type, market, self._mask)
      reference_curve = cashflow_streams.get_discount_curve(
          self._reference_curve_type, market, self._reference_mask)
      # Shape batch_shape + [1]
      daycount_fractions = tf.expand_dims(self._daycount_fractions, axis=-1)
      # Shape batch_shape + [1]
      fwd_rate = reference_curve.forward_rate(
          self._accrual_start_date.expand_dims(axis=-1),
          self._accrual_end_date.expand_dims(axis=-1),
          day_count_fraction=daycount_fractions)
      # Shape batch_shape + [1]
      discount_at_settlement = discount_curve.discount_factor(
          self._accrual_start_date.expand_dims(axis=-1))
      # Shape batch_shape + [1]
      discount_at_settlement = tf.where(daycount_fractions > 0.,
                                        discount_at_settlement,
                                        tf.zeros_like(discount_at_settlement))
      # Shape `batch_shape`
      discount_at_settlement = tf.squeeze(discount_at_settlement, axis=-1)
      fwd_rate = tf.squeeze(fwd_rate, axis=-1)
      return (self._short_position
              * discount_at_settlement
              * self._notional_amount * (fwd_rate - self._fixed_rate)
              * self._daycount_fractions / (1. + self._daycount_fractions
                                            * fwd_rate))

  @property
  def batch_shape(self) -> tf.Tensor:
    return self._batch_shape

  @property
  def names(self) -> tf.Tensor:
    """Returns a string tensor of names and instrument types.

    The shape of the output is  [batch_shape, 2].
    """
    return self._names


def _process_config(
    config: Union[ForwardRateAgreementConfig, Dict[str, Any], None]
    ) -> ForwardRateAgreementConfig:
  """Converts config to ForwardRateAgreementConfig."""
  if config is None:
    return ForwardRateAgreementConfig()
  if isinstance(config, ForwardRateAgreementConfig):
    return config
  model = config.get("model", "")
  discounting_curve = config.get("discounting_curve", dict())
  return ForwardRateAgreementConfig(discounting_curve=discounting_curve,
                                    model=model)


__all__ = ["ForwardRateAgreementConfig", "ForwardRateAgreement"]
