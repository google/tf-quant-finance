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
"""Rate indices."""

import enum
from typing import Optional, Union
import dataclasses
import tensorflow.compat.v2 as tf
from tf_quant_finance.experimental.pricing_platform.instrument_protos import rate_indices_pb2


RateIndexType = enum.Enum("RateIndexType",
                          zip(rate_indices_pb2.RateIndexType.keys(),
                              rate_indices_pb2.RateIndexType.keys()))
RateIndexType.__doc__ = "Supported rate indices."
RateIndexType.__repr__ = lambda self: self.value
RateIndexType.__call__ = lambda self: self.value


RateIndexName = enum.Enum("RateIndexName",
                          zip(rate_indices_pb2.RateIndexName.keys(),
                              rate_indices_pb2.RateIndexName.keys()))
RateIndexType.__doc__ = "Supported rate indices."
RateIndexType.__repr__ = lambda self: self.value
RateIndexType.__call__ = lambda self: self.value


@dataclasses.dataclass
class RateIndex:
  """Rate index object."""
  type: Union[RateIndexType, str]
  source: Optional[Union[str, tf.Tensor]] = ""
  name: Optional[Union[RateIndexName, tf.Tensor]] = ""

  def __post_init__(self):
    if isinstance(self.type, str):
      try:
        self.type = getattr(RateIndexType, self.type)
      except KeyError:
        raise ValueError(f"{self.type} is not a valid rate index type.")

  @classmethod
  def from_proto(cls, proto: rate_indices_pb2.RateIndex) -> "RateIndex":
    """Creates RateIndexType from a proto field value."""
    return cls(
        name=rate_indices_pb2.RateIndexName.Name(proto.name),
        type=rate_indices_pb2.RateIndexType.Name(proto.type),
        source=proto.source)


__all__ = ["RateIndex"]
