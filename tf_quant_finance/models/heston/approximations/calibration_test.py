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
"""Tests for Heston approximation calibration."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class CalibrationTest(parameterized.TestCase, tf.test.TestCase):

  def test_dynamic_shapes(self):
    """Test calibration function with dynamically shaped inputs."""
    dtype = np.float64

    observed_prices = np.array(
        [[29.45783648, 24.15306768, 19.73346361, 16.08792179, 13.10198113],
         [16.02509124, 21.51061818, 27.61195796, 34.24527161, 41.33562669]],
        dtype=dtype)

    strikes = np.array(
        [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=dtype)

    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    forwards = [[100.0], [100.0]]
    is_call_options = np.array([[True], [False]])

    volvol = np.array([0.75], dtype=dtype)
    initial_variance = np.array([0.25], dtype=dtype)
    mean_reversion = np.array([0.4], dtype=dtype)
    theta = np.array([0.9], dtype=dtype)

    @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=dtype),
                                  tf.TensorSpec([None, None], dtype=dtype),
                                  tf.TensorSpec([None, None], dtype=dtype),
                                  tf.TensorSpec([None, None], dtype=dtype),
                                  tf.TensorSpec([None, None], dtype=tf.bool),
                                  tf.TensorSpec([None], dtype=dtype),
                                  tf.TensorSpec([None], dtype=dtype),
                                  tf.TensorSpec([None], dtype=dtype),
                                  tf.TensorSpec([None], dtype=dtype)])
    def fn(observed_prices, strikes, expiries, forwards, is_call_options,
           initial_variance, mean_reversion, theta, volvol):
      models, is_converged, _ = tff.models.heston.calibration(
          prices=observed_prices,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call_options,
          mean_reversion=mean_reversion,
          initial_variance=initial_variance,
          theta=theta,
          volvol=volvol,
          rho=np.array(-0.5, dtype=dtype),
          optimizer_fn=tff.math.optimizer.bfgs_minimize,
          maximum_iterations=100)
      return models, is_converged

    models, is_converged = fn(
        observed_prices, strikes, expiries, forwards, is_call_options,
        initial_variance, mean_reversion, theta, volvol)
    [
        calibrated_initial_variance, calibrated_mean_reversion,
        calibrated_volvol, calibrated_rho, calibrated_theta, is_converged
    ] = self.evaluate([
        models.initial_variance[:, tf.newaxis],
        models.mean_reversion[:, tf.newaxis],
        models.volvol[:, tf.newaxis],
        models.rho[:, tf.newaxis], models.theta[:, tf.newaxis],
        is_converged
    ])
    with self.subTest('AllConverged'):
      self.assertTrue(all(is_converged))
    calibrated_prices = tff.models.heston.approximations.european_option_price(
        variances=calibrated_initial_variance,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        mean_reversion=calibrated_mean_reversion,
        theta=calibrated_theta,
        volvol=calibrated_volvol,
        rho=calibrated_rho,
        dtype=dtype)
    with self.subTest('PricesRecovered'):
      self.assertAllClose(calibrated_prices, observed_prices,
                          atol=1e-1, rtol=1e-3)

  def test_calibration_batch_limits(self):
    """Demonstrate that lower/upper limits can be set independently in batch."""
    dtype = np.float64

    observed_prices = np.array(
        [[29.45783648, 24.15306768, 19.73346361, 16.08792179, 13.10198113],
         [16.02509124, 21.51061818, 27.61195796, 34.24527161, 41.33562669]],
        dtype=dtype)

    strikes = np.array(
        [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=dtype)
    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    forwards = 100.0
    is_call_options = np.array([[True], [False]])

    models, is_converged, _ = tff.models.heston.calibration(
        prices=observed_prices,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        mean_reversion=np.array([0.4, 0.4], dtype=dtype),
        mean_reversion_lower_bound=np.array([0.05, 0.1]),
        mean_reversion_upper_bound=np.array([1.0, 5.0]),
        theta=np.array([0.9, 0.9], dtype=dtype),
        theta_lower_bound=np.array([0.1, 0.0]),
        theta_upper_bound=np.array([1.0, 1.0]),
        initial_variance=np.array([0.25, 0.25], dtype=dtype),
        initial_variance_lower_bound=np.array([0.1, 0.0]),
        initial_variance_upper_bound=np.array([5.0, 1.0]),
        volvol=np.array([0.75, 0.75], dtype=dtype),
        volvol_lower_bound=np.array([0.0, 0.05]),
        volvol_upper_bound=np.array([1.0, 5.0]),
        rho=np.array([-0.5, -0.5], dtype=dtype),
        rho_lower_bound=np.array([-1.0, -0.75]),
        rho_upper_bound=np.array([1.0, 0.75]),
        maximum_iterations=100)

    [
        calibrated_initial_variance, calibrated_mean_reversion,
        calibrated_volvol, calibrated_rho, calibrated_theta, is_converged
    ] = self.evaluate([
        models.initial_variance[:, tf.newaxis],
        models.mean_reversion[:, tf.newaxis],
        models.volvol[:, tf.newaxis],
        models.rho[:, tf.newaxis], models.theta[:, tf.newaxis],
        is_converged
    ])

    with self.subTest('AllConverged'):
      self.assertTrue(all(is_converged))

    calibrated_prices = tff.models.heston.approximations.european_option_price(
        variances=calibrated_initial_variance,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        mean_reversion=calibrated_mean_reversion,
        theta=calibrated_theta,
        volvol=calibrated_volvol,
        rho=calibrated_rho,
        dtype=dtype)
    with self.subTest('PricesRecovered'):
      self.assertAllClose(calibrated_prices, observed_prices,
                          atol=1e-3, rtol=1e-3)

  def test_spots_and_discounts(self):
    dtype = np.float64

    observed_prices = np.array(
        [[20.74264408, 13.0706132, 7.46141751, 3.8978837, 1.89302107],
         [3.45098237, 6.92500205, 11.82124538, 17.9901047, 25.19373215]],
        dtype=dtype)

    strikes = np.array(
        [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=dtype)
    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    is_call_options = np.array([[True], [False]])

    discount_factors = np.array([[0.99], [0.98]], dtype=dtype)
    spots = np.array([[99.0], [98.0]], dtype=dtype)

    models, is_converged, _ = tff.models.heston.calibration(
        prices=observed_prices,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        discount_factors=discount_factors,
        is_call_options=is_call_options,
        mean_reversion=np.array([0.1], dtype=dtype),
        theta=np.array([0.3], dtype=dtype),
        initial_variance=np.array([0.01], dtype=dtype),
        volvol=np.array([0.2], dtype=dtype),
        rho=np.array([0.0], dtype=dtype),
        optimizer_fn=tff.math.optimizer.bfgs_minimize,
        maximum_iterations=100)

    [
        calibrated_initial_variance, calibrated_mean_reversion,
        calibrated_volvol, calibrated_rho, calibrated_theta, is_converged
    ] = self.evaluate([
        models.initial_variance[:, tf.newaxis],
        models.mean_reversion[:, tf.newaxis],
        models.volvol[:, tf.newaxis],
        models.rho[:, tf.newaxis], models.theta[:, tf.newaxis],
        is_converged
    ])

    with self.subTest('AllConverged'):
      self.assertTrue(all(is_converged))

    calibrated_prices = tff.models.heston.approximations.european_option_price(
        variances=calibrated_initial_variance,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        discount_factors=discount_factors,
        is_call_options=is_call_options,
        mean_reversion=calibrated_mean_reversion,
        theta=calibrated_theta,
        volvol=calibrated_volvol,
        rho=calibrated_rho,
        dtype=dtype)
    with self.subTest('PricesRecovered'):
      self.assertAllClose(calibrated_prices, observed_prices,
                          atol=1e-3, rtol=1e-3)

  def test_validate_args(self):
    dtype = np.float64

    observed_prices = np.array(
        [[29.45783648, 24.15306768, 19.73346361, 16.08792179, 13.10198113],
         [16.02509124, 21.51061818, 27.61195796, 34.24527161, 41.33562669]],
        dtype=dtype)

    strikes = np.array(
        [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=dtype)
    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    forwards = 100.0
    is_call_options = np.array([[True], [False]])

    initial_variance = np.array([0.5], dtype=dtype)
    mean_reversion = np.array([0.25], dtype=dtype)
    theta = np.array([0.5], dtype=dtype)

    # Fails because `volvol` is outside the limits.
    with self.subTest('volvolInvalidArgumentCaught'):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        _, is_converged, _ = tff.models.heston.calibration(
            prices=observed_prices,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
            mean_reversion=mean_reversion,
            initial_variance=initial_variance,
            theta=theta,
            volvol=np.array([10.5], dtype=dtype),
            volvol_lower_bound=0.0,
            volvol_upper_bound=10.0,
            rho=np.array([0.0], dtype=dtype),
            rho_lower_bound=-0.75,
            rho_upper_bound=0.75,
            validate_args=True)
        self.evaluate(is_converged)

    # Fails because `rho` exceeds its expected bounds.
    with self.subTest('rhoInvalidArgumentCaught'):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        _, is_converged, _ = tff.models.heston.calibration(
            prices=observed_prices,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
            mean_reversion=mean_reversion,
            initial_variance=initial_variance,
            theta=theta,
            volvol=np.array([0.5], dtype=dtype),
            volvol_lower_bound=0.0,
            volvol_upper_bound=10.0,
            rho=np.array([0.76], dtype=dtype),
            rho_lower_bound=-0.75,
            rho_upper_bound=0.75,
            validate_args=True)
        self.evaluate(is_converged)

    # Arguments are okay.
    _, is_converged, _ = tff.models.heston.calibration(
        prices=observed_prices,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        mean_reversion=mean_reversion,
        initial_variance=initial_variance,
        theta=theta,
        volvol=np.array([0.5], dtype=dtype),
        volvol_lower_bound=0.0,
        volvol_upper_bound=10.0,
        rho=np.array([0.0, 0.0], dtype=dtype),
        rho_lower_bound=-0.75,
        rho_upper_bound=0.75,
        validate_args=True)
    is_converged = self.evaluate(is_converged)
    with self.subTest('allArgumentsWithinBounds'):
      self.assertTrue(all(is_converged))

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_noise_price_based',
          'true_mean_reversion': np.array([0.25, 0.25], dtype=np.float64),
          'true_initial_variance': np.array([0.5, 0.5], dtype=np.float64),
          'true_theta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.1, 0.1], dtype=np.float64),
          'true_rho': np.array([0.0, 0.0], dtype=np.float64),
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'low_noise_price_based',
          'true_mean_reversion': np.array([0.25, 0.25], dtype=np.float64),
          'true_initial_variance': np.array([0.5, 0.5], dtype=np.float64),
          'true_theta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.1, 0.1], dtype=np.float64),
          'true_rho': np.array([0.0, 0.0], dtype=np.float64),
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'noise_size': 0.01,
          'price_tol': (1e-2, 1e-2)
      },
  )
  def test_calibration(self, true_mean_reversion,
                       true_initial_variance, true_theta, true_volvol, true_rho,
                       max_iterations, tolerance, noise_size, price_tol):
    dtype = np.float64

    # Construct some market conditions.
    strikes = np.array([
        [80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
                       dtype=dtype)
    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    is_call_options = np.array([[True], [False]])
    forwards = 100.0

    # Generate some prices.
    denoised_prices = tff.models.heston.approximations.european_option_price(
        forwards=forwards,
        strikes=strikes,
        expiries=expiries,
        is_call_options=is_call_options,
        mean_reversion=np.expand_dims(true_mean_reversion, axis=-1),
        variances=np.expand_dims(true_initial_variance, axis=-1),
        theta=np.expand_dims(true_theta, axis=-1),
        volvol=np.expand_dims(true_volvol, axis=-1),
        rho=np.expand_dims(true_rho, axis=-1),
        dtype=dtype)

    # Add noise to the prices
    observed_prices = denoised_prices + tf.random.stateless_normal(
        denoised_prices.shape,
        stddev=noise_size * denoised_prices,
        seed=[2, 4],
        dtype=dtype)

    # Calibrate the models.
    models, _, _ = tff.models.heston.calibration(
        prices=observed_prices,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        mean_reversion=np.array([0.4, 0.4], dtype=dtype),
        initial_variance=np.array([0.25, 0.25], dtype=dtype),
        theta=np.array([0.9, 0.9], dtype=dtype),
        volvol=np.array([0.75, 0.75], dtype=dtype),
        rho=np.array([-0.5, -0.5], dtype=dtype),
        maximum_iterations=max_iterations,
        tolerance=tolerance,
        dtype=dtype)

    [
        calibrated_initial_variance, calibrated_mean_reversion,
        calibrated_volvol, calibrated_rho, calibrated_theta
    ] = self.evaluate([
        models.initial_variance, models.mean_reversion, models.volvol,
        models.rho, models.theta
    ])

    # Back out the approximated prices from the calibrated model, and check that
    # they agree with our input prices (up to noise)
    calibrated_prices = tff.models.heston.approximations.european_option_price(
        forwards=forwards,
        strikes=strikes,
        expiries=expiries,
        is_call_options=is_call_options,
        variances=np.array(
            np.expand_dims(calibrated_initial_variance, axis=1), dtype=dtype),
        mean_reversion=np.array(
            np.expand_dims(calibrated_mean_reversion, axis=1), dtype=dtype),
        theta=np.array(np.expand_dims(calibrated_theta, axis=1), dtype=dtype),
        volvol=np.array(np.expand_dims(calibrated_volvol, axis=1), dtype=dtype),
        rho=np.array(np.expand_dims(calibrated_rho, axis=1), dtype=dtype),
        dtype=dtype)

    calibrated_prices, denoised_prices = self.evaluate(
        [calibrated_prices, denoised_prices])
    with self.subTest('noiseScenarios'):
      self.assertAllClose(calibrated_prices, denoised_prices,
                          atol=price_tol[0], rtol=price_tol[1])

if __name__ == '__main__':
  tf.test.main()
