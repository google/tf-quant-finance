# Copyright 2021 Google LLC
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
"""Tests for variance_swaps."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tf_quant_finance.black_scholes import variance_swaps
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class VarianceSwapsTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for the variance_swaps module."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'Puts',
          'strikes': np.array([100.0, 90.0, 80.0, 70.0])
      }, {
          'testcase_name': 'Calls',
          'strikes': np.array([100.0, 110.0, 120.0, 130.0])
      })
  def test_replicating_weights(self, strikes):
    """Tests ability to match 'hand' calculated variance replicating weights."""
    reference_strikes = 100.0
    delta_strike = 10.0
    expiries = 1.0
    # This is the value of (A 4) in Demeterfi et al.
    payoff_values = 2 * ((strikes - reference_strikes) / reference_strikes -
                         np.log(strikes / reference_strikes))
    # This is the value of the ratio term in (A 7) in Demeterfi et al.
    slope_values = np.diff(payoff_values / delta_strike)
    # Literal calculation of (A 7/8) for all weights. The library uses
    # first differences rather than cumsums for efficiency due to algebra.
    expected_weights = []
    for v in slope_values:
      expected_weights.append(v - np.sum(expected_weights))
    weights = self.evaluate(
        variance_swaps.replicating_weights(
            strikes, reference_strikes, expiries, dtype=tf.float64))
    self.assertAllClose(weights, expected_weights, 1e-6)

  def test_replicating_weights_supports_batching(self):
    put_strikes = tf.constant([[100, 95, 90, 85]], dtype=np.float64)
    batch_put_strikes = batch_put_strikes = tf.concat(
        [put_strikes, put_strikes, 2 * (put_strikes - 100) + 100], axis=0)
    batch_reference = tf.math.reduce_max(batch_put_strikes, axis=1)
    batch_expiries = tf.constant([0.25, 0.5, 0.25], dtype=tf.float64)
    expected_shape = np.array(batch_put_strikes.shape)
    expected_shape[-1] = expected_shape[-1] - 1
    batch_weights = self.evaluate(
        variance_swaps.replicating_weights(
            batch_put_strikes,
            batch_reference,
            batch_expiries))
    self.assertAllEqual(batch_weights.shape, expected_shape)
    for i in range(3):
      row_weights = self.evaluate(
          variance_swaps.replicating_weights(
              batch_put_strikes[i, :], batch_reference[i], batch_expiries[i]))
      self.assertAllEqual(row_weights, batch_weights[i, :])

  def test_replicating_weights_raises_validation_error(self):
    strikes = np.array([1, 2, 3, 2, 1])
    reference_strike = 3
    expiry = 1
    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = self.evaluate(
          variance_swaps.replicating_weights(
              strikes,
              reference_strike,
              expiry,
              validate_args=True,
              dtype=tf.float64))

  @parameterized.named_parameters({
      'testcase_name':
          'Demeterfi_et_al',
      'call_strikes':
          np.array([100., 105., 110., 115., 120., 125., 130., 135., 140.]),
      'call_weights':
          np.array([19.63, 36.83, 33.55, 30.69, 28.19, 25.98, 24.02, 22.27]),
      'call_volatilities':
          np.array([0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, np.nan]),
      'put_strikes':
          np.array(
              [100., 95., 90., 85., 80., 75., 70., 65., 60., 55., 50., 45.]),
      'put_weights':
          np.array([
              20.98, 45., 50.15, 56.23, 63.49, 72.26, 82.98, 96.27, 113.05,
              134.63, 163.04
          ]),
      'put_volatilities':
          np.array([
              0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
              np.nan
          ]),
      'reference_strikes':
          100.0,
      'expiries':
          0.25,
      'discount_rates':
          0.05,
      # Paper rounds to 2 dp in places (and variably within columns elsewhere)
      'tolerance':
          1e-2,
      'k_var':
          0.20467**2,  # Paper works on % scale.
  })
  def test_variance_swap_demeterfi_example(self, call_strikes, call_weights,
                                           call_volatilities, put_strikes,
                                           put_weights, put_volatilities,
                                           reference_strikes, expiries,
                                           discount_rates, tolerance, k_var):
    """Tests ability to match 'hand' calculated variance replicating weights."""
    # Paper quotes weights inflated to forward values.
    discount_factor = np.exp(discount_rates * expiries)
    calculated_call_weights = self.evaluate(
        variance_swaps.replicating_weights(
            call_strikes, reference_strikes, expiries, dtype=tf.float64))
    matched_call_weights = discount_factor * 100.0**2 * calculated_call_weights
    self.assertAllClose(matched_call_weights, call_weights, tolerance)
    calculated_put_weights = self.evaluate(
        variance_swaps.replicating_weights(
            put_strikes, reference_strikes, expiries, dtype=tf.float64))
    matched_put_weights = discount_factor * 100.0**2 * calculated_put_weights
    self.assertAllClose(matched_put_weights, put_weights, tolerance)
    variance_price = self.evaluate(
        tff.black_scholes.variance_swap_fair_strike(
            put_strikes,
            put_volatilities,
            call_strikes,
            call_volatilities,
            expiries,
            discount_rates,
            reference_strikes,
            reference_strikes,
            dtype=tf.float64))
    self.assertAllClose(variance_price, k_var, 1e-2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_validation',
          'validate_args': True
      }, {
          'testcase_name': 'without_validation',
          'validate_args': False
      })
  def test_variance_swap_fair_strike_supports_batching(self, validate_args):
    dtype = tf.float64
    batch_call_strikes = tf.repeat(
        tf.expand_dims(tf.range(100, 120, 5, dtype=dtype), 0), 3, axis=0)
    batch_put_strikes = tf.repeat(
        tf.expand_dims(tf.range(100, 80, -5, dtype=dtype), 0), 3, axis=0)
    batch_vols = 0.2 * tf.ones((3, 4), dtype=dtype)
    batch_shape = (3,)
    reference_strikes = 100.0 * tf.ones(batch_shape, dtype=dtype)
    batch_expiries = tf.constant([0.25, 0.5, 1.0], dtype=dtype)
    discount_rates = 0.05 * tf.ones(batch_shape, dtype=dtype)
    batch_variance_price = self.evaluate(
        tff.black_scholes.variance_swap_fair_strike(
            batch_put_strikes,
            batch_vols,
            batch_call_strikes,
            batch_vols,
            batch_expiries,
            discount_rates,
            reference_strikes,
            reference_strikes,
            validate_args=validate_args,
            dtype=dtype))

    self.assertEqual(batch_variance_price.shape, batch_shape)
    for i in range(3):
      row_variance_price = self.evaluate(
          tff.black_scholes.variance_swap_fair_strike(
              batch_put_strikes[i, :],
              batch_vols[i, :],
              batch_call_strikes[i, :],
              batch_vols[i, :],
              batch_expiries[i],
              discount_rates[i],
              reference_strikes[i],
              reference_strikes[i],
              dtype=tf.float64))
      self.assertAllEqual(row_variance_price, batch_variance_price[i])

  def test_variance_swap_fair_strike_raises_validation_error(self):
    dtype = tf.float64
    # Mismatching shapes for strikes and vols.
    strikes = tf.ones((3, 2), dtype=dtype)
    vols = tf.ones((3, 4), dtype=dtype)
    reference_strike = 1.0
    discount_rate = 0.0
    expiry = 1.0
    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = self.evaluate(
          tff.black_scholes.variance_swap_fair_strike(
              strikes,
              vols,
              strikes,
              vols,
              expiry,
              discount_rate,
              reference_strike,
              reference_strike,
              validate_args=True,
              dtype=dtype))


if __name__ == '__main__':
  tf.test.main()
