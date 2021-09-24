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
"""European swaptions."""
import dataclasses

from typing import Any, Dict, List, Optional, Union

import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance.experimental.pricing_platform.framework.core import instrument
from tf_quant_finance.experimental.pricing_platform.framework.core import models
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import cashflow_streams
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import interest_rate_swap
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.swaption import proto_utils
from tf_quant_finance.experimental.pricing_platform.instrument_protos import swaption_pb2 as swaption_proto
from tf_quant_finance.models import hull_white


@dataclasses.dataclass(frozen=True)
class SwaptionConfig:
  # TODO(b/167921989): Make model_params part of market data.
  model_params: models.HullWhite1FactorConfig
  model: models.InterestRateModelType = None


class Swaption(instrument.Instrument):
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
  and calculating its price using the Hull-White One Factor Model.

  ```python
  import tensorflow_quant_finance as tff
  core = tff.experimental.pricing_platform.framework.core
  dates = tff.datetime
  market_data = tff.experimental.pricing_platform.framework.market_data
  rates_instruments =
  tff.experimental.pricing_platform.framework.rate_instruments

  instrument_protos = tff.experimental.pricing_platform.instrument_protos
  date_pb2 = instrument_protos.date
  decimal_pb2 = instrument_protos.decimal
  swap_pb2 = instrument_protos.interest_rate_swap
  period_pb2 = instrument_protos.period
  rate_indices_pb2 = instrument_protos.rate_indices
  swaption_pb2 = instrument_protos.swaption

  swapproto = swap_pb2.InterestRateSwap(
      effective_date=date_pb2.Date(year=2022, month=2, day=10),
      maturity_date=date_pb2.Date(year=2025, month=2, day=10),
      currency=core.currencies.Currency.USD(),
      pay_leg=swap_pb2.SwapLeg(
          fixed_leg=swap_pb2.FixedLeg(
              currency=core.currencies.Currency.USD(),
              coupon_frequency=period_pb2.Period(type='MONTH', amount=6),
              notional_amount=decimal_pb2.Decimal(units=100),
              fixed_rate=decimal_pb2.Decimal(nanos=11000000),
              daycount_convention=core.daycount_conventions
              .DayCountConventions.ACTUAL_365(),
              business_day_convention=core.business_days.BusinessDayConvention
              .NO_ADJUSTMENT(),
              settlement_days=0)),
      receive_leg=swap_pb2.SwapLeg(
          floating_leg=swap_pb2.FloatingLeg(
              currency=core.currencies.Currency.USD(),
              coupon_frequency=period_pb2.Period(type='MONTH', amount=3),
              reset_frequency=period_pb2.Period(type='MONTH', amount=3),
              notional_amount=decimal_pb2.Decimal(units=100),
              floating_rate_type=rate_indices_pb2.RateIndex(type='LIBOR_3M'),
              daycount_convention=core.daycount_conventions
              .DayCountConventions.ACTUAL_365(),
              business_day_convention=core.business_days.BusinessDayConvention
              .NO_ADJUSTMENT(),
              settlement_days=0)))
  swaption_proto = swaption_pb2.Swaption(
      swap=swapproto, expiry_date=date_pb2.Date(year=2022, month=2, day=10))

  hull_white_config = core.models.HullWhite1FactorConfig(
        mean_reversion=[0.03], volatility=[0.01])
  config = rates_instruments.swaption.SwaptionConfig(
      model_params=hull_white_config)

  curve_dates = self._valuation_date + dates.years(
      [1, 2, 3, 5, 7, 10, 30])

  curve_discounts = np.exp(-0.01 * np.array([1, 2, 3, 5, 7, 10, 30]))

    dates = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
             [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
    discounts = [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
                 0.72494879, 0.37602059]
    libor_3m_config = {
        "interpolation_method": interpolation_method.InterpolationMethod.LINEAR
    }
    self._market_data_dict = {
        "rates": {
            "USD": {
                "risk_free_curve": {
                    "dates": dates,
                    "discounts": discounts,
                },
                "LIBOR_3M": {
                    "dates": dates,
                    "discounts": discounts,
                    "config": libor_3m_config,
                }
            }
        },
        "reference_date": [(2020, 2, 10)],
    }

  market = market_data.MarketDataDict(market_data_dict)

  swaption = rates_instruments.swaption.Swaption.from_protos(
      [swaption_proto], config=config)
  price = swaption[0].price(market)

  # Expected result: 1.38594754
  ```

  ### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

  def __init__(self,
               swap: interest_rate_swap.InterestRateSwap,
               expiry_date: Optional[Union[dateslib.DateTensor,
                                           List[List[int]]]],
               config: SwaptionConfig,
               batch_names: Optional[tf.Tensor] = None,
               dtype: Optional[types.Dtype] = None,
               name: Optional[str] = None):
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
      config: An input of type `SwaptionConfig` specifying the
        necessary information for swaption valuation.
      batch_names: A string `Tensor` of instrument names. Should be of shape
        `batch_shape + [2]` specying name and instrument type. This is useful
        when the `from_protos` method is used and the user needs to identify
        which instruments got batched together.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the Swaption object or created by the Swaption
        object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'swaption'.
    """
    self._name = name or "swaption"

    with tf.name_scope(self._name):
      if batch_names is not None:
        self._names = tf.convert_to_tensor(batch_names,
                                           name="batch_names")
      else:
        self._names = None
      self._dtype = dtype or tf.float64
      self._expiry_date = dateslib.convert_to_date_tensor(expiry_date)
      self._swap = swap
      self._config = config

  def price(self,
            market: pmd.ProcessedMarketData,
            name: Optional[str] = None):
    """Returns the present value of the swaption on the valuation date.

    Args:
      market: A instance of type `ProcessedMarketData` which contains the
        necessary information for pricing the swaption.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A Rank `Tensor` of shape `batch_shape` containing the modeled price of
      each  Swaption contract based on the input market data.

    Raises:
      ValueError: If an unsupported model is supplied to the function.
    """
    model = (self._config.model or
             models.InterestRateModelType.HULL_WHITE_ONE_FACTOR)
    name = name or (self._name + "_price")
    with tf.name_scope(name):
      valuation_date = dateslib.convert_to_date_tensor(market.date)
      strike = self._swap.fixed_rate()

      expiry_time = dateslib.daycount_actual_365_fixed(
          start_date=valuation_date,
          end_date=self._expiry_date,
          dtype=self._dtype)

      if model == models.InterestRateModelType.HULL_WHITE_ONE_FACTOR:
        option_value = self._price_hull_white_1_factor(
            valuation_date, market, strike, expiry_time)
      else:
        raise ValueError("Unsupported model.")

      return option_value

  @classmethod
  def create_constructor_args(
      cls, proto_list: List[swaption_proto.Swaption],
      config: SwaptionConfig = None) -> Dict[str, Any]:
    """Creates a dictionary to initialize Swaption."""
    raise NotImplementedError("`create_constructor_args` not yet implemented "
                              "for Swaption instrument.")

  @classmethod
  def from_protos(
      cls,
      proto_list: List[swaption_proto.Swaption],
      config: SwaptionConfig = None) -> List["Swaption"]:
    prepare_swaptions = proto_utils.from_protos(proto_list, config)
    instruments = []
    for kwargs in prepare_swaptions.values():
      kwargs["swap"] = interest_rate_swap.InterestRateSwap.from_protos(
          kwargs["swap"])[0]
      instruments.append(cls(**kwargs))
    return instruments

  @classmethod
  def group_protos(
      cls,
      proto_list: List[swaption_proto.Swaption],
      config: SwaptionConfig = None
      ) -> Dict[str, List["Swaption"]]:
    return proto_utils.group_protos(proto_list, config)

  def batch_shape(self) -> types.StringTensor:
    """Returns batch shape of the instrument."""
    pass

  def names(self) -> types.StringTensor:
    """Returns a string tensor of names and instrument types.

    The shape of the output is  [batch_shape, 2].
    """
    pass

  def _price_hull_white_1_factor(self, valuation_date, market,
                                 strike, expiry_time):
    """Price the swaption using Hull-White 1-factor model."""

    if isinstance(
        self._swap.pay_leg(), cashflow_streams.FloatingCashflowStream):
      floating_leg = self._swap.pay_leg()
      fixed_leg = self._swap.receive_leg()
    else:
      fixed_leg = self._swap.pay_leg()
      floating_leg = self._swap.receive_leg()

    # Get the reference curve from the floating leg of the underlying swap
    reference_curve = market.yield_curve(floating_leg.reference_curve_type[0])
    valuation_date_ordinal = tf.cast(valuation_date.ordinal(),
                                     dtype=self._dtype)

    def _refercence_rate_fn(t):
      # The input `t` is a real `Tensor` specifying the time from valuation.
      # We convert it into a `DateTensor` by first conversting it into the
      # corresponding ordinal (assuming ACT_365 convention).
      interpolation_ordinals = tf.cast(
          tf.round(t * 365.0 +  valuation_date_ordinal), dtype=tf.int32)
      interpolation_dates = dateslib.convert_to_date_tensor(
          interpolation_ordinals)
      return reference_curve.discount_rate(interpolation_dates)

    floating_leg_start_times = dateslib.daycount_actual_365_fixed(
        start_date=valuation_date,
        end_date=floating_leg.coupon_start_dates,
        dtype=self._dtype)
    floating_leg_end_times = dateslib.daycount_actual_365_fixed(
        start_date=valuation_date,
        end_date=floating_leg.coupon_end_dates,
        dtype=self._dtype)
    fixed_leg_payment_times = dateslib.daycount_actual_365_fixed(
        start_date=valuation_date,
        end_date=fixed_leg.cashflow_dates,
        dtype=self._dtype)
    # Add the extra dimension corresponding to multiple payments in the fixed
    # leg.
    fixed_leg_coupon = tf.broadcast_to(tf.expand_dims(strike, axis=-1),
                                       fixed_leg_payment_times.shape)
    is_payer_swaption = tf.convert_to_tensor(
        isinstance(self._swap.pay_leg(), cashflow_streams.FixedCashflowStream),
        dtype=tf.bool)
    notional = self._swap.pay_leg().notional

    hw_price = hull_white.swaption_price(
        expiries=expiry_time,
        floating_leg_start_times=floating_leg_start_times,
        floating_leg_end_times=floating_leg_end_times,
        fixed_leg_payment_times=fixed_leg_payment_times,
        floating_leg_daycount_fractions=floating_leg.daycount_fractions,
        fixed_leg_daycount_fractions=fixed_leg.daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=_refercence_rate_fn,
        is_payer_swaption=is_payer_swaption,
        use_analytic_pricing=True,
        notional=notional,
        mean_reversion=self._config.model_params.mean_reversion,
        volatility=self._config.model_params.volatility,
        dtype=self._dtype)
    return hw_price


__all__ = ["Swaption", "SwaptionConfig"]
