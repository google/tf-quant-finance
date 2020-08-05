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
"""Interest rate swap."""

import copy
from typing import List, Dict, Optional, Union

import dataclasses
import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance import math as tff_math
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import instrument
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils as market_data_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import cashflow_streams
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import coupon_specs
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.interest_rate_swap import proto_utils
from tf_quant_finance.experimental.pricing_platform.instrument_protos import interest_rate_swap_pb2 as ir_swap


@dataclasses.dataclass(frozen=True)
class InterestRateSwapConfig:
  discounting_curve: Dict[types.CurrencyProtoType, curve_types.CurveType]
  model: str


class InterestRateSwap(instrument.Instrument):
  """Represents a batch of Interest Rate Swaps (IRS).

  An Interest rate swap (IRS) is a contract between two counterparties for an
  exchange of a series of payments over a period of time. The payments are made
  periodically (for example quarterly or semi-annually) where the last payment
  is made at the maturity (or termination) of the contract. In the case of
  fixed-for-floating IRS, one counterparty pays a fixed rate while the other
  counterparty's payments are linked to a floating index, most commonly the
  LIBOR rate. On the other hand, in the case of interest rate basis swap, the
  payments of both counterparties are linked to a floating index. Typically, the
  floating rate is observed (or fixed) at the beginning of each period while the
  payments are made at the end of each period [1].

  For example, consider a vanilla swap with the starting date T_0 and maturity
  date T_n and equally spaced coupon payment dates T_1, T_2, ..., T_n such that

  T_0 < T_1 < T_2 < ... < T_n and dt_i = T_(i+1) - T_i    (A)

  The floating rate is fixed on T_0, T_1, ..., T_(n-1) and both the fixed and
  floating payments are made on T_1, T_2, ..., T_n (payment dates).

  The InterestRateSwap class can be used to create and price multiple IRS
  simultaneously. The class supports vanilla fixed-for-floating swaps as
  well as basis swaps. However all IRS within an IRS object must be priced using
  a common reference and discount curve.

  #### Example (non batch):
  The following example illustrates the construction of an IRS instrument and
  calculating its price.

  ```python
  DayCountConventions = daycount_conventions.DayCountConventions
  BusinessDayConvention = business_days.BusinessDayConvention
  RateIndex = instrument_protos.rate_indices.RateIndex
  Currency = currencies.Currency

  swap = ir_swap.InterestRateSwap(
      effective_date=date_pb2.Date(year=2020, month=2, day=2),
      maturity_date=date_pb2.Date(year=2023, month=2, day=2),
      currency=Currency.USD(),
      pay_leg=ir_swap.SwapLeg(
          fixed_leg=ir_swap.FixedLeg(
              currency=Currency.USD(),
              coupon_frequency=period_pb2.Period(type="MONTH", amount=6),
              notional_amount=decimal_pb2.Decimal(units=10000),
              fixed_rate=decimal_pb2.Decimal(nanos=31340000),
              daycount_convention=DayCountConventions.ACTUAL_365(),
              business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
              settlement_days=0)),
      receive_leg=ir_swap.SwapLeg(
          floating_leg=ir_swap.FloatingLeg(
              currency=Currency.USD(),
              coupon_frequency=period_pb2.Period(type="MONTH", amount=3),
              reset_frequency=period_pb2.Period(type="MONTH", amount=3),
              notional_amount=decimal_pb2.Decimal(units=10000),
              floating_rate_type=RateIndex(type="LIBOR_3M"),
              daycount_convention=DayCountConventions.ACTUAL_365(),
              business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
              settlement_days=0)))
  market_data_dict = {"USD": {
      "risk_free_curve":
      {"date": [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
                [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]],
        "discount": [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
                     0.72494879, 0.37602059]},
      "LIBOR_3M":
      {"date": [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
                [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]],
        "discount": [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
                     0.72494879, 0.37602059]}}}

  ois_config = market_data_config.RateConfig(
      interpolation_method=interpolation_method.InterpolationMethod.LINEAR)
  libor_3m_config = market_data_config.RateConfig(
      interpolation_method=interpolation_method.InterpolationMethod.LINEAR)
  rate_config = {"USD": {"risk_free_curve": ois_config,
                         "LIBOR_3M": libor_3m_config}}
  valuation_date = [(2020, 2, 8)]
  market = market_data.MarketDataDict(valuation_date, market_data_dict,
                                      config=rate_config)
  swaps = interest_rate_swap.InterestRateSwap.from_protos([swap])
  swaps[0].price(market)
  # Expected: [-69.42497649]
  ```

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

  def __init__(self,
               start_date: Union[dateslib.DateTensor, List[List[int]]],
               maturity_date: Union[dateslib.DateTensor, List[List[int]]],
               pay_leg: Union[coupon_specs.FixedCouponSpecs,
                              coupon_specs.FloatCouponSpecs],
               receive_leg: Union[coupon_specs.FixedCouponSpecs,
                                  coupon_specs.FloatCouponSpecs],
               swap_config: InterestRateSwapConfig = None,
               batch_names: Optional[tf.Tensor] = None,
               dtype: Optional[types.Dtype] = None,
               name: Optional[str] = None):
    """Initializes a batch of IRS contracts.

    Args:
      start_date: A `DateTensor` of `batch_shape` specifying the dates for the
        inception (start of the accrual) of the swap contracts. `batch_shape`
        corresponds to the number of instruments being created.
      maturity_date: A `DateTensor` broadcastable with `start_date` specifying
        the maturity dates for each contract.
      pay_leg: An instance of `FixedCouponSpecs` or `FloatCouponSpecs`
        specifying the coupon payments for the payment leg of the swap.
      receive_leg: An instance of `FixedCouponSpecs` or `FloatCouponSpecs`
        specifying the coupon payments for the receiving leg of the swap.
      swap_config: Optional `InterestRateSwapConfig`.
      batch_names: A string `Tensor` of instrument names. Should be of shape
        `batch_shape + [2]` specying name and instrument type. This is useful
        when the `from_protos` method is used and the user needs to identify
        which instruments got batched together.
      dtype: `tf.Dtype` of the input and output real `Tensor`s.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'interest_rate_swap'.
    """
    self._name = name or "interest_rate_swap"

    with tf.name_scope(self._name):
      if batch_names is not None:
        self._names = tf.convert_to_tensor(batch_names,
                                           name="batch_names")
      else:
        self._names = None
      self._dtype = dtype or tf.float64
      self._model = None  # Ignore

      currency = pay_leg.currency
      if pay_leg.currency != receive_leg.currency:
        raise ValueError("Pay and receive legs should have the same currency")
      if swap_config is not None:
        try:
          self._discount_curve_type = swap_config.discounting_curve[currency]
        except KeyError:
          risk_free = curve_types.RiskFreeCurve(currency=currency)
          self._discount_curve_type = risk_free
      else:
        # Default discounting is the risk free curve
        risk_free = curve_types.RiskFreeCurve(currency=currency)
        self._discount_curve_type = risk_free
      self._start_date = dateslib.convert_to_date_tensor(start_date)
      self._maturity_date = dateslib.convert_to_date_tensor(maturity_date)
      self._pay_leg = self._setup_leg(pay_leg)
      self._receive_leg = self._setup_leg(receive_leg)
      if self._receive_leg.batch_shape != self._pay_leg.batch_shape:
        raise ValueError("Batch shapes of pay and receive legs should be the "
                         "same.")
      self._batch_shape = self._pay_leg.batch_shape

  @classmethod
  def from_protos(
      cls, proto_list: List[ir_swap.InterestRateSwap],
      swap_config: InterestRateSwapConfig = None) -> List["InterestRateSwap"]:
    proto_dict = proto_utils.from_protos(proto_list, swap_config)
    intruments = []
    for kwargs in proto_dict.values():
      intruments.append(cls(**kwargs))
    return intruments

  @classmethod
  def group_protos(
      cls,
      proto_list: List[ir_swap.InterestRateSwap],
      fra_config: InterestRateSwapConfig = None
      ) -> Dict[str, List["InterestRateSwap"]]:
    return proto_utils.group_protos(proto_list, fra_config)

  def price(self,
            market: pmd.ProcessedMarketData,
            name: Optional[str] = None):
    """Returns the present value of the stream on the valuation date.

    Args:
      market: An instance of `ProcessedMarketData`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A `Tensor` of shape `batch_shape`  containing the modeled price of each
      IRS contract based on the input market data.
    """
    name = name or (self._name + "_price")
    with tf.name_scope(name):
      pay_cf = self._pay_leg.price(market)
      receive_cf = self._receive_leg.price(market)
      return receive_cf - pay_cf

  def annuity(self, market):
    """Returns the annuity of each swap on the vauation date."""
    return self._annuity(market)

  def par_rate(self, market):
    """Returns the par swap rate for the swap."""
    swap_annuity = self._annuity(market)
    if isinstance(self._pay_leg, cashflow_streams.FloatingCashflowStream):
      floating_leg = self._pay_leg
    else:
      floating_leg = self._receive_leg
    float_pv = floating_leg.price(market)

    return float_pv / swap_annuity / floating_leg.notional

  @property
  def batch_shape(self) -> tf.Tensor:
    return self._batch_shape

  @property
  def names(self) -> tf.Tensor:
    """Returns a string tensor of names and instrument types.

    The shape of the output is  [batch_shape, 2].
    """
    return self._names

  def pay_leg(self) -> Union[cashflow_streams.FloatingCashflowStream,
                             cashflow_streams.FixedCashflowStream]:
    """Returs pay leg cahsflow stream object."""
    return self._pay_leg

  def receive_leg(self) -> Union[cashflow_streams.FloatingCashflowStream,
                                 cashflow_streams.FixedCashflowStream]:
    """Receive pay leg cahsflow stream object."""
    return self._receive_leg

  def ir_delta(self,
               tenor: types.DateTensor,
               processed_market_data: pmd.ProcessedMarketData,
               curve_type: curve_types.CurveType,
               shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes delta wrt to the tenor perturbation."""
    raise NotImplementedError("Coming soon.")

  # TODO(b/160672068): Add `ir_delta_parallel` to cashflow streams.
  def ir_delta_parallel_leg(
      self,
      leg: Union[cashflow_streams.FloatingCashflowStream,
                 cashflow_streams.FixedCashflowStream],
      processed_market_data: pmd.ProcessedMarketData,
      curve_type: Optional[curve_types.CurveType] = None,
      shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes delta wrt to the curve parallel perturbation for a leg."""
    # TODO(b/160671927): Find a better way to update market data entries.
    market_bumped = copy.copy(processed_market_data)
    reference_curve = None
    reference_curve_type = None
    if curve_type is None:
      # Extract discount and reference curves
      curve_type = leg.discount_curve_type
      if isinstance(leg, cashflow_streams.FloatingCashflowStream):
        reference_curve_type = leg.reference_curve_type
        reference_curve = processed_market_data.yield_curve(
            reference_curve_type)

    discount_curve = processed_market_data.yield_curve(curve_type)
    # IR delta is the sensitivity wrt the yield perturbation
    yields, times = market_data_utils.get_yield_and_time(
        discount_curve, processed_market_data.date, self._dtype)
    # Extract yields for reference curve, if needed
    if reference_curve is not None:
      reference_yields, reference_times = market_data_utils.get_yield_and_time(
          reference_curve, processed_market_data.date, self._dtype)
    def price_fn(bump):
      """Prices the leg with a given bump."""
      discount_factors = (1 + yields + bump)**(-times)
      discount_curve.set_discount_factor_nodes(discount_factors)
      def _discount_curve_fn(input_curve_type):
        """Updates discount curve."""
        if input_curve_type == curve_type:
          return discount_curve
        elif input_curve_type == reference_curve_type:
          reference_discount_factors = (
              1 + reference_yields + bump)**(-reference_times)
          reference_curve.set_discount_factor_nodes(reference_discount_factors)
          return reference_curve
        else:
          return processed_market_data.yield_curve(curve_type)
      market_bumped.yield_curve = _discount_curve_fn
      return leg.price(market_bumped)
    if shock_size is None:
      bump = tf.constant(0, dtype=self._dtype,
                         name="bump")
      return tff_math.fwd_gradient(price_fn, bump)
    shock_size = tf.convert_to_tensor(shock_size, dtype=self._dtype,
                                      name="shock_size")
    price_no_bump = leg.price(processed_market_data)
    price_with_bump = price_fn(shock_size)
    delta = (price_with_bump - price_no_bump) / shock_size
    return delta

  def ir_delta_parallel(
      self,
      processed_market_data: pmd.ProcessedMarketData,
      curve_type: Optional[curve_types.CurveType] = None,
      shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes delta wrt to the curve parallel perturbation."""
    # IR delta is the sensitivity wrt the yield perturpation
    delta_pay_leg = self.ir_delta_parallel_leg(
        self._pay_leg, processed_market_data, curve_type, shock_size)
    delta_receive_leg = self.ir_delta_parallel_leg(
        self._receive_leg, processed_market_data, curve_type, shock_size)
    return delta_receive_leg - delta_pay_leg

  def ir_vega(self,
              tenor: types.DateTensor,
              processed_market_data: pmd.ProcessedMarketData,
              shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes vega wrt to the tenor perturbation."""
    raise NotImplementedError("Not supported for InterestRateSwap.")

  def _setup_leg(
      self,
      leg: Union[coupon_specs.FixedCouponSpecs, coupon_specs.FloatCouponSpecs]
      ) -> Union[cashflow_streams.FixedCashflowStream,
                 cashflow_streams.FloatingCashflowStream]:
    """Setup swap legs."""
    if isinstance(leg, coupon_specs.FixedCouponSpecs):
      return cashflow_streams.FixedCashflowStream(
          start_date=self._start_date,
          end_date=self._maturity_date,
          coupon_spec=leg,
          discount_curve_type=self._discount_curve_type,
          dtype=self._dtype)
    elif isinstance(leg, coupon_specs.FloatCouponSpecs):
      return cashflow_streams.FloatingCashflowStream(
          start_date=self._start_date,
          end_date=self._maturity_date,
          coupon_spec=leg,
          discount_curve_type=self._discount_curve_type,
          dtype=self._dtype)
    else:
      raise ValueError(f"Unknown leg type {type(leg)}")

  def _annuity(self, market: pmd.ProcessedMarketData) -> tf.Tensor:
    """Returns the annuity of each swap on the vauation date."""
    num_fixed_legs = 0
    if isinstance(self._pay_leg, cashflow_streams.FixedCashflowStream):
      fixed_leg = self._pay_leg
      num_fixed_legs += 1
    if isinstance(self._receive_leg, cashflow_streams.FixedCashflowStream):
      fixed_leg = self._receive_leg
      num_fixed_legs += 1
    if num_fixed_legs == 0:
      raise ValueError("Swap does not have a fixed leg.")
    if num_fixed_legs == 2:
      raise ValueError("Swap should not have both fixed leg.")
    discount_curve = market.yield_curve(self._discount_curve_type)
    discount_factors = discount_curve.discount_factor(
        fixed_leg.cashflow_dates)
    return tf.math.reduce_sum(
        discount_factors * fixed_leg.daycount_fractions, axis=-1)


__all__ = ["InterestRateSwapConfig", "InterestRateSwap"]
