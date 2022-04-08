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
from typing import Any, Dict, List

import tensorflow.compat.v2 as tf

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
  def create_constructor_args(
      cls, proto_list: List[types.ProtobufBaseType],
      **kwargs) -> Dict[str, Any]:
    """Creates a dictionary to initialize the Instrument.

    The output dictionary is such that the instruments can be initialized
    as follows:
    ```
    initializer = create_constructor_args(proto_list, **kwargs)
    instruments = [Instrument(**data) for data in initializer.values()]
    ```

    The keys of the output dictionary are unique identifiers of the batched
    instruments. This is useful for identifying an existing graph that could be
    reused for the instruments without the need of rebuilding the graph.

    Args:
      proto_list: A list of protos for which the initialization arguments are
        constructed.
      **kwargs: Any other keyword args needed by an implementation.

    Returns:
      A possibly nested dictionary such that each value provides initialization
      arguments for the Instrument.
    """
    del proto_list, kwargs
    return dict()

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


__all__ = ["Instrument"]
