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
"""Instrument interface."""

import abc
from typing import Any, Dict, List, Optional

import tensorflow.compat.v2 as tf

from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types


class Instrument(abc.ABC):
  """Instrument interface."""

  @classmethod
  def from_protos(
      cls,
      proto_list: List[types.ProtobufBaseType],
      **kwargs) -> List["Instrument"]:
    """Converts a list of protos to a list of batched `Instruments`."""
    del proto_list, kwargs
    return []

  @classmethod
  def from_dict(cls,
                dict_list: List[Dict[str, Any]],
                **kwargs) -> List["Instrument"]:
    """Converts a list of dict messages to a list of batched `Instruments`."""
    del dict_list, kwargs
    return []

  @classmethod
  def group_protos(
      cls,
      proto_list: List[types.ProtobufBaseType],
      **kwargs) -> Dict[str, List[types.ProtobufBaseType]]:
    """Creates a dict of batchable protos.

    For a list of protos, generates a dictionary `{key: grouped_protos}` such
    that the `grouped_protos` can be batched together.

    Args:
      proto_list: A list of `Instrument` protos.
      **kwargs: Any extra arguments. E.g., pricing configuration.

    Returns:
      A dictionary of grouped protos.
    """
    del proto_list, kwargs
    return []

  @abc.abstractproperty
  def batch_shape(self) -> types.StringTensor:
    """Returns batch shape of the instrument."""
    pass

  @abc.abstractproperty
  def names(self) -> types.StringTensor:
    """Returns a string tensor of names and instrument types.

    The shape of the output is  [batch_shape, 2].
    """
    pass

  @abc.abstractmethod
  def price(self, processed_market_data: pmd.ProcessedMarketData) -> tf.Tensor:
    """Computes price of the batch of the instrument against the market data."""
    pass

  @abc.abstractmethod
  def ir_delta(self,
               tenor: types.DateTensor,
               processed_market_data: pmd.ProcessedMarketData,
               curve_type: Optional[curve_types.CurveType] = None,
               shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes delta wrt to the tenor perturbation."""
    pass

  @abc.abstractmethod
  def ir_delta_parallel(
      self,
      processed_market_data: pmd.ProcessedMarketData,
      curve_type: Optional[curve_types.CurveType] = None,
      shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes delta wrt to the curve parallel perturbation."""
    pass

  @abc.abstractmethod
  def ir_vega(self,
              tenor: types.DateTensor,
              processed_market_data: pmd.ProcessedMarketData,
              shock_size: Optional[float] = None) -> tf.Tensor:
    """Computes vega wrt to the implied volatility perturbation."""
    pass


__all__ = ["Instrument"]
