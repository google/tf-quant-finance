# Copyright 2019 Google LLC
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
"""Tests for implied_vol_approx."""


from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

bs = tff.black_scholes


@test_util.run_all_in_graph_and_eager_modes
class ImpliedVolTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods in implied_vol module."""

  def test_implied_vol(self):
    """Basic test of the implied vol calculation."""
    np.random.seed(6589)
    n = 100
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      volatilities = np.exp(np.random.randn(n) / 2)
      forwards = np.exp(np.random.randn(n))
      strikes = forwards * (1 + (np.random.rand(n) - 0.5) * 0.2)
      expiries = np.exp(np.random.randn(n))
      prices = self.evaluate(
          bs.option_price(
              volatilities=volatilities,
              strikes=strikes,
              expiries=expiries,
              forwards=forwards,
              dtype=dtype))
      # Using the default method
      implied_vols_default = self.evaluate(
          bs.implied_vol(
              prices=prices,
              strikes=strikes,
              expiries=expiries,
              forwards=forwards,
              dtype=dtype))
      self.assertArrayNear(volatilities, implied_vols_default, 0.2)

      # Using Newton explicitly
      implied_vols_newton = self.evaluate(
          bs.implied_vol(
              prices=prices,
              strikes=strikes,
              expiries=expiries,
              forwards=forwards,
              dtype=dtype,
              method=bs.ImpliedVolMethod.NEWTON))
      self.assertArrayNear(implied_vols_default, implied_vols_newton, 1e-5)

      # Using approximation.
      implied_vols_approx = self.evaluate(
          bs.implied_vol(
              prices=prices,
              strikes=strikes,
              expiries=expiries,
              forwards=forwards,
              dtype=dtype,
              method=bs.ImpliedVolMethod.FAST_APPROX))
      self.assertArrayNear(volatilities, implied_vols_approx, 0.6)

  def test_validate(self):
    """Test the algorithm doesn't raise where it shouldn't."""
    np.random.seed(6589)
    n = 100
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      volatilities = np.exp(np.random.randn(n) / 2)
      forwards = np.exp(np.random.randn(n))
      strikes = forwards * (1 + (np.random.rand(n) - 0.5) * 0.2)
      expiries = np.exp(np.random.randn(n))
      prices = self.evaluate(
          bs.option_price(
              volatilities=volatilities,
              strikes=strikes,
              expiries=expiries,
              forwards=forwards,
              dtype=dtype))

      implied_vols = self.evaluate(
          bs.implied_vol(
              prices=prices,
              strikes=strikes,
              expiries=expiries,
              forwards=forwards,
              validate_args=True,
              dtype=dtype))
      self.assertArrayNear(volatilities, implied_vols, 0.6)

  @parameterized.named_parameters(
      # This case should hit the call lower bound since C = F - K.
      ('call_lower', 0.0, 1.0, 1.0, 1.0, True),
      # This case should hit the call upper bound since C = F
      ('call_upper', 1.0, 1.0, 1.0, 1.0, True),
      # This case should hit the put upper bound since C = K
      ('put_lower', 1.0, 1.0, 1.0, 1.0, False),
      # This case should hit the call lower bound since C = F - K.
      ('put_upper', 0.0, 1.0, 1.0, 1.0, False))
  def test_validate_raises(self, price, forward, strike, expiry,
                           is_call_option):
    """Test algorithm raises appropriately."""
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      prices = np.array([price]).astype(dtype)
      forwards = np.array([forward]).astype(dtype)
      strikes = np.array([strike]).astype(dtype)
      expiries = np.array([expiry]).astype(dtype)
      is_call_options = np.array([is_call_option])
      with self.assertRaises(tf.errors.InvalidArgumentError):
        implied_vols = bs.implied_vol(
            prices=prices,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
            validate_args=True,
            dtype=dtype)
        self.evaluate(implied_vols)


if __name__ == '__main__':
  tf.test.main()
