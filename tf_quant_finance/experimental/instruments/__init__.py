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
"""Instruments."""

from tf_quant_finance.experimental.instruments import bond
from tf_quant_finance.experimental.instruments import cap_floor
from tf_quant_finance.experimental.instruments import cashflow_stream
from tf_quant_finance.experimental.instruments import cms_swap
from tf_quant_finance.experimental.instruments import eurodollar_futures
from tf_quant_finance.experimental.instruments import floating_rate_note
from tf_quant_finance.experimental.instruments import forward_rate_agreement
from tf_quant_finance.experimental.instruments import interest_rate_swap
from tf_quant_finance.experimental.instruments import overnight_index_linked_futures
from tf_quant_finance.experimental.instruments import rate_curve
from tf_quant_finance.experimental.instruments import rates_common
from tf_quant_finance.experimental.instruments import swaption
from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

Bond = bond.Bond
CapAndFloor = cap_floor.CapAndFloor
EurodollarFutures = eurodollar_futures.EurodollarFutures
Swaption = swaption.Swaption
OvernightIndexLinkedFutures = overnight_index_linked_futures.OvernightIndexLinkedFutures
ForwardRateAgreement = forward_rate_agreement.ForwardRateAgreement
FloatingRateNote = floating_rate_note.FloatingRateNote
RateCurve = rate_curve.RateCurve
InterestRateMarket = rates_common.InterestRateMarket
DayCountConvention = rates_common.DayCountConvention
FixedCashflowStream = cashflow_stream.FixedCashflowStream
FloatingCashflowStream = cashflow_stream.FloatingCashflowStream
CMSCashflowStream = cms_swap.CMSCashflowStream
CMSSwap = cms_swap.CMSSwap
InterestRateSwap = interest_rate_swap.InterestRateSwap
FixedCouponSpecs = rates_common.FixedCouponSpecs
FloatCouponSpecs = rates_common.FloatCouponSpecs
CMSCouponSpecs = rates_common.CMSCouponSpecs
AverageType = rates_common.AverageType
InterestRateModelType = rates_common.InterestRateModelType
ratecurve_from_discounting_function = rate_curve.ratecurve_from_discounting_function

_allowed_symbols = [
    'Bond',
    'CapAndFloor',
    'CMSCashflowStream',
    'CMSCouponSpecs',
    'CMSSwap',
    'EurodollarFutures',
    'FloatingRateNote',
    'ForwardRateAgreement',
    'OvernightIndexLinkedFutures',
    'RateCurve',
    'Swaption',
    'InterestRateMarket',
    'InterestRateModelType',
    'DayCountConvention',
    'FixedCashflowStream',
    'FloatingCashflowStream',
    'InterestRateSwap',
    'FixedCouponSpecs',
    'FloatCouponSpecs',
    'AverageType',
    'ratecurve_from_discounting_function',
]

remove_undocumented(__name__, _allowed_symbols)
