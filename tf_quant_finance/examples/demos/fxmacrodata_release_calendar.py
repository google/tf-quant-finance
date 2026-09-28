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
"""FXMacroData demos for TensorFlow Quant Finance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow.compat.v2 as tf

try:
    from urllib.parse import urlencode
    from urllib.request import Request
    from urllib.request import urlopen
except ImportError:  # pragma: no cover
    from urllib import urlencode
    from urllib2 import Request
    from urllib2 import urlopen

FXMACRODATA_BASE_URL = "https://api.fxmacrodata.com/v1"
FXMACRODATA_API_KEY_ENV_VARS = ("FXMACRODATA_API_KEY", "FXMD_API_KEY")
# List endpoints return at most 100 rows per request, newest first.
FXMACRODATA_PAGE_SIZE = 100
FXMACRODATA_MAX_PAGES = 1000


def _env_api_key():
    for name in FXMACRODATA_API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def fetch_fxmacrodata(path, params=None, api_key=None, base_url=FXMACRODATA_BASE_URL):
    """Fetches a raw FXMacroData read endpoint payload."""
    query = {k: v for k, v in (params or {}).items() if v is not None}
    headers = {"User-Agent": "fxmacrodata-tfq-demo"}
    token = api_key or _env_api_key()
    if token:
        headers["X-API-Key"] = token
    url = "%s/%s" % (base_url.rstrip("/"), path.lstrip("/"))
    if query:
        url = "%s?%s" % (url, urlencode(query))
    request = Request(url, headers=headers)
    response = urlopen(request, timeout=30)  # nosec B310
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()


def fetch_fxmacrodata_rows(
    path, params=None, limit=None, api_key=None, base_url=FXMACRODATA_BASE_URL
):
    """Fetches up to `limit` rows (all rows if None) from a paginated endpoint."""
    rows = []
    offset = 0
    for _ in range(FXMACRODATA_MAX_PAGES):
        page_size = FXMACRODATA_PAGE_SIZE
        if limit is not None:
            page_size = min(page_size, int(limit) - len(rows))
        if page_size <= 0:
            break
        query = dict(params or {}, limit=page_size, offset=offset)
        payload = fetch_fxmacrodata(path, query, api_key=api_key, base_url=base_url)
        page = payload_rows(payload)
        rows.extend(page)
        pagination = payload.get("pagination") if isinstance(payload, dict) else None
        if not page or not pagination or not pagination.get("has_more"):
            break
        offset = pagination.get("next_offset") or offset + len(page)
    return rows


def payload_rows(payload):
    data = payload.get("data", []) if isinstance(payload, dict) else payload
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [
            dict({"indicator": key}, **value)
            if isinstance(value, dict)
            else {"indicator": key, "value": value}
            for key, value in data.items()
        ]
    return []


def fetch_event_time_tensor(
    currency="usd",
    limit=100,
    min_tier=None,
    api_key=None,
    base_url=FXMACRODATA_BASE_URL,
):
    """Fetches release-calendar event times as a TensorFlow int64 tensor."""
    payload = fetch_fxmacrodata(
        "calendar/%s" % currency.lower(), api_key=api_key, base_url=base_url
    )
    rows = payload_rows(payload)
    if min_tier is not None:
        rows = [
            row for row in rows if int(row.get("market_tier") or 99) <= int(min_tier)
        ]
    rows = rows[: max(1, int(limit))]
    event_seconds = [
        int(row["announcement_datetime"])
        for row in rows
        if row.get("announcement_datetime") is not None
    ]
    return tf.constant(event_seconds, dtype=tf.int64), rows


def fetch_macro_value_tensor(
    currency,
    indicator,
    start_date=None,
    end_date=None,
    limit=100,
    api_key=None,
    base_url=FXMACRODATA_BASE_URL,
):
    """Fetches macro announcement values as a TensorFlow float64 tensor."""
    rows = fetch_fxmacrodata_rows(
        "announcements/%s/%s" % (currency.lower(), indicator),
        {"start_date": start_date, "end_date": end_date},
        limit=limit,
        api_key=api_key,
        base_url=base_url,
    )
    values = [
        float(row.get("actual", row.get("value", row.get("val"))))
        for row in rows
        if row.get("actual", row.get("value", row.get("val"))) is not None
    ]
    return tf.constant(values, dtype=tf.float64), rows


def fetch_forex_rate_tensor(
    base,
    quote,
    start_date=None,
    end_date=None,
    limit=100,
    api_key=None,
    base_url=FXMACRODATA_BASE_URL,
):
    """Fetches FX spot rates as a TensorFlow float64 tensor."""
    rows = fetch_fxmacrodata_rows(
        "forex/%s/%s" % (base.lower(), quote.lower()),
        {"start_date": start_date, "end_date": end_date},
        limit=limit,
        api_key=api_key,
        base_url=base_url,
    )
    values = [
        float(row.get("value", row.get("val", row.get("rate"))))
        for row in rows
        if row.get("value", row.get("val", row.get("rate"))) is not None
    ]
    return tf.constant(values, dtype=tf.float64), rows


def fetch_prediction_tensor(
    currency,
    indicator,
    start_date=None,
    end_date=None,
    limit=100,
    api_key=None,
    base_url=FXMACRODATA_BASE_URL,
):
    """Fetches prediction rows and extracts numeric predictions when present."""
    rows = fetch_fxmacrodata_rows(
        "predictions/%s/%s" % (currency.lower(), indicator),
        {"start_date": start_date, "end_date": end_date},
        limit=limit,
        api_key=api_key,
        base_url=base_url,
    )
    values = []
    for row in rows:
        value = row.get("predicted_value", row.get("prediction", row.get("forecast")))
        if value is not None:
            values.append(float(value))
    return tf.constant(values, dtype=tf.float64), rows


if __name__ == "__main__":
    event_times, events = fetch_event_time_tensor(currency="usd", limit=10, min_tier=2)
    print("Loaded %d FXMacroData events" % len(events))
    print(event_times)
