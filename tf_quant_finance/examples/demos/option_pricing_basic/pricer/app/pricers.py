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
"""Computes option prices using TensorFlow Finance."""

import datetime
import time

import numpy as np
import tensorflow as tf
import tf_quant_finance as tff


class Timer:
  """A simple timer."""

  def __init__(self):
    self.start_time = 0
    self.end_time = 0

  def __enter__(self) -> "Timer":
    self.start_time = time.time()
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    del unused_type, unused_value, unused_traceback
    self.end_time = time.time()

  @property
  def elapsed_ms(self) -> float:
    """Returns the elapsed time in milliseconds."""
    return (self.end_time - self.start_time) * 1000


def _price(spot_mkt, vol_mkt, rate_mkt, underliers, strikes, call_put_flag,
           expiry_ordinals):
  """Prices the options."""
  # Get mkt data for each option
  spots = tf.gather(spot_mkt, underliers)
  vols = tf.gather(vol_mkt, underliers)
  rates = tf.gather(rate_mkt, underliers)
  # Convert expiries into time.
  expiry_ordinals = tf.cast(expiry_ordinals, dtype=tf.int32)
  expiry_dates = tff.datetime.dates_from_ordinals(expiry_ordinals)
  pricing_date = tff.datetime.dates_from_datetimes([datetime.date.today()])
  expiry_times = tff.datetime.daycount_actual_360(
      start_date=pricing_date, end_date=expiry_dates, dtype=np.float64)
  prices = tff.black_scholes.option_price(
      volatilities=vols,
      strikes=strikes,
      expiries=expiry_times,
      spots=spots,
      discount_rates=rates,
      is_call_options=call_put_flag)
  return prices


class TffOptionPricer:
  """Prices options using TFF."""

  def __init__(self, batch_size=1000000, num_assets=1000):
    dtype = np.float64
    self._pricer = tf.function(_price)
    # Do a warm-up. This initializes a bunch of stuff which improves performance
    # at request serving time.
    if batch_size is not None and num_assets is not None:
      self._pricer(
          np.zeros([num_assets], dtype=dtype),  # spot_mkt
          np.zeros([num_assets], dtype=dtype),  # vol_mkt
          np.zeros([num_assets], dtype=dtype),  # rate_mkt
          np.zeros([batch_size], dtype=np.int32),  # underliers
          np.zeros([batch_size], dtype=dtype),  # strikes
          np.zeros([batch_size], dtype=np.bool),  # call_put_flag
          np.ones([batch_size], dtype=np.int32))  # expiry_ordinals

  def price(self, spot_mkt, vol_mkt, rate_mkt, underliers, strikes,
            call_put_flag, expiry_ordinals):
    """Prices options."""
    prices = self._pricer(spot_mkt, vol_mkt, rate_mkt, underliers, strikes,
                          call_put_flag, expiry_ordinals)
    return prices
