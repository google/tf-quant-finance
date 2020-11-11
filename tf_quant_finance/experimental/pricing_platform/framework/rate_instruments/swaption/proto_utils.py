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
"""Utilities for creating a Swaption instrument."""

from typing import Any, Dict, List, Tuple

from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.interest_rate_swap import proto_utils as swap_proto_utils
from tf_quant_finance.experimental.pricing_platform.instrument_protos import swaption_pb2 as swaption_proto


def get_hash(swaption_instance: swaption_proto.Swaption) -> Tuple[int, bool]:
  """Computes hash key for the batching strategy."""
  return swap_proto_utils.get_hash(swaption_instance.swap)


def from_protos(
    proto_list: List[swaption_proto.Swaption],
    config: "SwaptionConfig" = None
    ) -> Dict[str, Any]:
  """Creates a dictionary of preprocessed swap data."""
  prepare_swaptions = {}
  for swaption_instance in proto_list:
    swap_instance = swaption_instance.swap
    expiry_date = swaption_instance.expiry_date

    # For swaptions lets use the same keys as the underlying swaps.
    h, _ = get_hash(swaption_instance)
    name = swaption_instance.metadata.id
    instrument_type = swaption_instance.metadata.instrument_type
    if h in prepare_swaptions:
      prepare_swaptions[h]["expiry_date"].append(expiry_date)
      prepare_swaptions[h]["swap"].append(swap_instance)
      prepare_swaptions[h]["batch_names"].append([name, instrument_type])
    else:
      prepare_swaptions[h] = {
          "expiry_date": [expiry_date],
          "swap": [swap_instance],
          "config": config,
          "batch_names": [[name, instrument_type]]
      }
  return prepare_swaptions


def group_protos(
    proto_list: List[swaption_proto.Swaption],
    config: "SwaptionConfig" = None
    ) -> Dict[str, Any]:
  """Creates a dictionary of grouped protos."""
  del config

  grouped_swaptions = {}
  for swaption_instance in proto_list:
    h, _ = get_hash(swaption_instance)
    if h in grouped_swaptions:
      grouped_swaptions[h].append(swaption_instance)
    else:
      grouped_swaptions[h] = [swaption_instance]
  return grouped_swaptions
