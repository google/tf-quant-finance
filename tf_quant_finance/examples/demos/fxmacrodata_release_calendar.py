# Copyright 2026 Google LLC
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
# ==============================================================================
"""FXMacroData release-calendar demo for TensorFlow Quant Finance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow.compat.v2 as tf

try:
  from urllib.parse import urlencode
  from urllib.request import urlopen
except ImportError:  # pragma: no cover
  from urllib import urlencode
  from urllib2 import urlopen

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def fetch_event_time_tensor(currency="usd", limit=100, min_tier=None,
                            api_key=None, base_url=FXMACRODATA_BASE_URL):
  """Fetches FXMacroData release times as a TensorFlow int64 tensor."""
  limit = max(1, int(limit))
  params = {"limit": limit}
  token = api_key or os.environ.get("FXMACRODATA_API_KEY")
  if token:
    params["api_key"] = token

  url = "%s/calendar/%s?%s" % (
      base_url.rstrip("/"), currency.lower(), urlencode(params))
  response = urlopen(url, timeout=30)  # nosec B310
  try:
    payload = json.loads(response.read().decode("utf-8"))
  finally:
    response.close()

  rows = payload.get("data", [])
  if min_tier is not None:
    rows = [
        row for row in rows
        if int(row.get("market_tier") or 99) <= int(min_tier)
    ]

  event_seconds = [
      int(row["announcement_datetime"])
      for row in rows[:limit]
      if row.get("announcement_datetime") is not None
  ]
  return tf.constant(event_seconds, dtype=tf.int64), rows[:limit]


if __name__ == "__main__":
  event_times, events = fetch_event_time_tensor(currency="usd", limit=10,
                                                min_tier=2)
  print("Loaded %d FXMacroData events" % len(events))
  print(event_times)
