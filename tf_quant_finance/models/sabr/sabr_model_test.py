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
"""Tests for SABR Model."""

import math
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.models import euler_sampling

SabrModel = tff.models.SabrModel


@test_util.run_all_in_graph_and_eager_modes
class SabrModelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("zero_beta", 0., 2., 0., 0.1),
      ("non_zero_beta", 0.5, 1., 0., 0.1),
      ("one_beta", 1., 0.5, 0., 0.1, 1., 1.),
      ("correlated_process", 0.5, 1, 0.5, 0.1),
      ("fallback_to_euler", lambda *args: tf.constant(0.5, dtype=tf.float64), 1,
       0.5, 0.01))
  def test_volatility(self,
                      beta,
                      volvol,
                      rho,
                      time_step,
                      initial_forward=1.,
                      initial_volatility=0.1):
    """Tests that volatility follows a log-normal distribution."""
    dtype = tf.float64
    times = [0.1, 0.3, 0.5]
    num_samples = 1000
    test_seed = [123, 124]

    process = SabrModel(
        beta=beta,
        volvol=volvol,
        rho=rho,
        dtype=dtype,
        enable_unbiased_sampling=True)
    paths = process.sample_paths(
        initial_forward=initial_forward,
        initial_volatility=initial_volatility,
        times=times,
        time_step=time_step,
        num_samples=num_samples,
        seed=test_seed,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)

    paths = self.evaluate(paths)
    for i in range(len(times)):
      time = times[i]
      path_slice = paths[:, i, 1]
      # Our current implemenation of absorbing bondary conditions leaves one
      # non-positive value, then zeros out future values. Ignore any negative
      # values here for now. However, make sure we're not losing too many
      # samples to the absorbing boundary.
      path_slice = path_slice[path_slice > 0]
      self.assertGreaterEqual(
          len(path_slice), num_samples * 0.9, msg="Too many invalid samples")
      mean = np.mean(np.log(path_slice))
      stddev = np.std(np.log(path_slice))
      self.assertAllClose((-0.5 * volvol**2 * time, volvol * math.sqrt(time)),
                          (mean - math.log(initial_volatility), stddev),
                          rtol=0.1,
                          atol=0.1)

  @parameterized.named_parameters(
      ("non_zero_beta", 0.5, 1., 0., 0.1),
      ("correlated_process", 0.5, 1., 0.5, 0.1),
      ("fallback_to_euler", lambda *args: tf.constant(0.5, dtype=tf.float64), 1,
       0.5, 0.01))
  def test_drift(self, beta, volvol, rho, time_step):
    """Tests E[F(t)] == F_0."""
    dtype = tf.float64
    times = [0.1, 1.0]
    num_samples = 10
    initial_forward, initial_volatility = 1., 0.1
    test_seed = [123, 124]

    process = SabrModel(
        beta=beta,
        volvol=volvol,
        rho=rho,
        dtype=dtype,
        enable_unbiased_sampling=True)
    paths = process.sample_paths(
        initial_forward=initial_forward,
        initial_volatility=initial_volatility,
        times=times,
        time_step=time_step,
        num_samples=num_samples,
        seed=test_seed,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)

    paths = self.evaluate(paths)
    for i in range(len(times)):
      mean = np.mean(paths[:, i, 0])
      self.assertAllClose(mean, initial_forward, rtol=0.1, atol=0.1)

  @parameterized.named_parameters(
      ("beta_too_small", -1, 1, 0, [1.]), ("beta_too_large", 1.1, 1, 0, [0.5]),
      ("negative_volvol", 0.5, -1, 0, [1.]),
      ("rho_too_small", 0.5, 1, -2, [1.]), ("rho_too_large", 0.5, 1, 2, [0.5]),
      ("times_not_increasing", 0.5, 1, 0, [2., 1.]))
  def test_sabr_model_validate_raises_error(self, beta, volvol, rho, times):
    """Test that the SABR model raises errors appropriately."""
    dtype = np.float64
    time_step = 0.1
    num_samples = 10
    initial_forward, initial_volatility = 1., 1.
    test_seed = 123
    with self.assertRaises(tf.errors.InvalidArgumentError):
      process = SabrModel(
          beta=beta,
          volvol=volvol,
          rho=rho,
          dtype=dtype,
          enable_unbiased_sampling=True)
      paths = process.sample_paths(
          initial_forward=initial_forward,
          initial_volatility=initial_volatility,
          times=times,
          time_step=time_step,
          num_samples=num_samples,
          seed=test_seed,
          validate_args=True)
      paths = self.evaluate(paths)

  def test_sabr_model_absorbing_boundary(self):
    """Test that the vol function gets zeroed out for absorbing boundary."""
    process = SabrModel(
        beta=lambda *args: tf.constant(0.5, dtype=tf.float64),
        volvol=1.,
        rho=0.,
        dtype=tf.float64,
        enable_unbiased_sampling=True)
    vol_matrix = process.volatility_fn()(
        t=0., x=tf.constant([0., 1.], dtype=tf.float64))
    self.assertAllEqual(vol_matrix, tf.zeros([2, 2], dtype=tf.float64))

  @parameterized.named_parameters(
      ("initial_test_put", 0.8, 1., 0.2, 0.01, 100., [100.], 0.12),
      ("initial_test_put2", 0.5, 1., 0.2, 0.01, 100., [100.], 0.12),
      ("initial_test_put3", 0., 1., 0.2, 0.01, 100., [100.], 0.12),
      ("initial_test_call", 0.8, 1., 0.2, 0.01, 100., [100.], 0.12, False),
      ("reference_1_test_4", 0.4, 0.8, -0.6, 0.01, 0.07, [0.07], 0.4, False),
      ("ir_test_put1", 0.8, 0.5, 0.2, 0.01, 0.01, [0.009, 0.005], 0.75),
      ("ir_test_put2", 0.8, 0.5, 0.2, 0.01, 0.01, [0.009, 0.005], 2.51),
      ("ir_test_call1", 0.8, 0.5, 0.2, 0.01, 0.01, [0.01, 0.015, 0.02], 0.75,
       False), ("ir_test_call2", 0.8, 0.5, 0.2, 0.01, 0.01, [0.01, 0.015, 0.02
                                                            ], 2.51, False))
  def test_pricing_european_option(self,
                                   beta,
                                   volvol,
                                   rho,
                                   time_step,
                                   initial_forward,
                                   strikes,
                                   initial_volatility,
                                   put_option=True):
    """Test that the SABR model computes the same price as the Euler method."""
    dtype = np.float64
    times = [0.5]
    num_samples = 10000
    test_seed = [123, 124]
    beta = tf.convert_to_tensor(beta, dtype=dtype)
    volvol = tf.convert_to_tensor(volvol, dtype=dtype)
    rho = tf.convert_to_tensor(rho, dtype=dtype)

    if put_option:
      option_fn = lambda samples, strike: strike - samples
    else:
      option_fn = lambda samples, strike: samples - strike

    drift_fn = lambda _, x: tf.zeros_like(x)

    def _vol_fn(t, x):
      """The volatility function for the SABR model."""
      del t
      f = x[..., 0]
      v = x[..., 1]
      fb = f**beta
      m11 = v * fb * tf.math.sqrt(1 - tf.square(rho))
      m12 = v * fb * rho
      m21 = tf.zeros_like(m11)
      m22 = volvol * v
      mc1 = tf.concat([tf.expand_dims(m11, -1), tf.expand_dims(m21, -1)], -1)
      mc2 = tf.concat([tf.expand_dims(m12, -1), tf.expand_dims(m22, -1)], -1)
      # Set up absorbing boundary.
      should_be_zero = tf.expand_dims(
          tf.expand_dims((beta != 0) & (f <= 0.), -1), -1)
      vol_matrix = tf.concat([tf.expand_dims(mc1, -1),
                              tf.expand_dims(mc2, -1)], -1)
      return tf.where(should_be_zero, tf.zeros_like(vol_matrix), vol_matrix)

    euler_paths = euler_sampling.sample(
        dim=2,
        drift_fn=drift_fn,
        volatility_fn=_vol_fn,
        times=times,
        time_step=time_step,
        num_samples=num_samples,
        initial_state=[initial_forward, initial_volatility],
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=test_seed,
        dtype=dtype)

    euler_paths = self.evaluate(euler_paths)
    euler_samples = euler_paths[..., 0]

    process = SabrModel(
        beta=beta,
        volvol=volvol,
        rho=rho,
        dtype=dtype,
        enable_unbiased_sampling=True)
    # Use a 10x grid step to make the test faster.
    paths = process.sample_paths(
        initial_forward=initial_forward,
        initial_volatility=initial_volatility,
        times=times,
        time_step=time_step * 10,
        num_samples=num_samples,
        seed=test_seed,
        validate_args=True,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)
    paths = self.evaluate(paths)
    samples = paths[..., 0]

    for strike in strikes:
      euler_mean, euler_price = (np.average(euler_samples),
                                 np.average(
                                     np.maximum(
                                         option_fn(euler_samples, strike), 0)))
      mean, price = (np.average(samples),
                     np.average(np.maximum(option_fn(samples, strike), 0)))
      self.assertAllClose([euler_mean, euler_price], [mean, price],
                          rtol=0.05,
                          atol=0.05)

  def test_relative_error(self):
    """Replicate tests from reference [1] test case 4."""
    dtype = np.float64
    num_samples = 1000
    test_seed = [123, 124]

    initial_forward = tf.constant(0.07, dtype=dtype)
    initial_volatility = tf.constant(0.4, dtype=dtype)
    volvol = tf.constant(0.8, dtype=dtype)
    beta = tf.constant(0.4, dtype=dtype)
    rho = tf.constant(-0.6, dtype=dtype)
    times = [1.0]
    timesteps = [0.0625, 0.03125]
    strike = 0.4
    process = SabrModel(
        beta=beta,
        volvol=volvol,
        rho=rho,
        dtype=dtype,
        enable_unbiased_sampling=True)
    process_table = []

    # We compute the relative error across time steps ts for a fixed expiry T:
    # error = | C(T, ts_i) - C(S(T, ts_{i+1})) |
    # where ts_i > ts_{i+1} and C( .. ) is the call option price
    for ts in timesteps:
      paths = process.sample_paths(
          initial_forward=initial_forward,
          initial_volatility=initial_volatility,
          times=times,
          time_step=ts,
          num_samples=num_samples,
          seed=test_seed,
          validate_args=True,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)
      paths = self.evaluate(paths)
      samples = paths[..., 0]
      price = np.average(np.maximum(samples - strike, 0))
      process_table.append(price)

    euler_error = 0.0001681610489796333
    process_error = 0
    for i in range(0, len(timesteps) - 1):
      process_error += np.abs(process_table[i] - process_table[i + 1])
    # Average relative error should be lower. Euler error is precomputed
    # for `test_seed` using STATELESS_ANTITHETIC random type.
    self.assertLessEqual(process_error, euler_error)

  @parameterized.named_parameters(
      ("zero_beta", 0.0, 2.0, 0.0, 0.1, 1.0, 0.1, True),
      ("non_zero_beta", 0.5, 1.0, 0.0, 0.1, 1.0, 0.1, True),
      ("one_beta", 1.0, 0.5, 0.0, 0.1, 1.0, 1.0, True),
      ("correlated_process", 0.5, 1, 0.5, 0.1, 1.0, 0.0, True),
      ("fallback_to_euler", lambda *args: tf.constant(0.5, dtype=tf.float64), 1,
       0.5, 0.01, 1.0, 0.0, True),
      ("nonzero_shift_unbiased_sampling_true", 0.5, 1.0, 0.0, 0.1, -2.0, 5.0,
       True),
      ("nonzero_shift_unbiased_sampling_false", 0.5, 1.0, 0.0, 0.1, -2.0, 5.0,
       False),
  )
  def test_forward_obeys_lower_bound(self,
                                     beta,
                                     volvol,
                                     rho,
                                     time_step,
                                     initial_forward,
                                     shift,
                                     enabled_unbiased_sampling,
                                     initial_volatility=0.1):
    """Tests that the forwards obey the lower bound."""
    dtype = tf.float64
    times = [0.1, 0.3, 0.5]
    num_samples = 1000
    test_seed = [123, 124]

    process = SabrModel(
        beta=beta,
        volvol=volvol,
        rho=rho,
        dtype=dtype,
        shift=shift,
        enable_unbiased_sampling=enabled_unbiased_sampling)
    paths = process.sample_paths(
        initial_forward=initial_forward,
        initial_volatility=initial_volatility,
        times=times,
        time_step=time_step,
        num_samples=num_samples,
        seed=test_seed,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)

    paths = self.evaluate(paths)
    forwards_axis = 0

    for i in range(len(times)):
      forwards_slice = paths[:, i, forwards_axis]
      self.assertTrue(np.all(forwards_slice >= -shift))

  @parameterized.named_parameters(
      ("nonzero_shift_unbiased_sampling_true", 0.5, 1.0, 0.0, 0.1, -1.0, 5.0,
       True),
      ("nonzero_shift_unbiased_sampling_false", 0.5, 1.0, 0.0, 0.1, -1.0, 5.0,
       False),
  )
  def test_negative_forward_rates_okay(self,
                                       beta,
                                       volvol,
                                       rho,
                                       time_step,
                                       initial_forward,
                                       shift,
                                       enabled_unbiased_sampling,
                                       initial_volatility=0.1):
    """Tests that the forwards can potentially be negative."""
    dtype = tf.float64
    times = [0.1, 0.3, 0.5]
    num_samples = 1000
    test_seed = [123, 124]

    process = SabrModel(
        beta=beta,
        volvol=volvol,
        rho=rho,
        dtype=dtype,
        shift=shift,
        enable_unbiased_sampling=enabled_unbiased_sampling)
    paths = process.sample_paths(
        initial_forward=initial_forward,
        initial_volatility=initial_volatility,
        times=times,
        time_step=time_step,
        num_samples=num_samples,
        seed=test_seed,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)

    paths = self.evaluate(paths)
    forwards_axis = 0

    # Though not strictly guaranteed, it's extremely unlikely that we won't
    # encounter *any* negatively-valued forward rates.
    forwards_slices = paths[:, :, forwards_axis]
    self.assertTrue(np.any(forwards_slices < 0))


if __name__ == "__main__":
  tf.test.main()
