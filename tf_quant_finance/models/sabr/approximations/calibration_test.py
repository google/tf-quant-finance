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
"""Tests for Sabr approximation calibration."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

# Helper aliases.
NORMAL = tff.models.sabr.approximations.SabrImpliedVolatilityType.NORMAL
LOGNORMAL = tff.models.sabr.approximations.SabrImpliedVolatilityType.LOGNORMAL


@test_util.run_all_in_graph_and_eager_modes
class CalibrationTest(parameterized.TestCase, tf.test.TestCase):

  def test_calibration_docstring(self):
    """Test the example in the docstring.

    In this example, we are calibrating a SABR model using the lognormal
    volatility approximation for implied volatility, and we explicitly fix the
    beta's ourselves.
    """
    dtype = np.float64

    observed_prices = np.array(
        [[20.09689284, 10.91953054, 4.25012702, 1.11561839, 0.20815853],
         [3.34813209, 6.03578711, 10.2874194, 16.26824328, 23.73850935]],
        dtype=dtype)

    strikes = np.array(
        [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=dtype)
    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    forwards = 100.0
    is_call_options = np.array([[True], [False]])

    beta = np.array([0.5, 0.5], dtype=dtype)

    models, is_converged, _ = tff.models.sabr.calibration(
        prices=observed_prices,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        beta=beta,
        calibrate_beta=False,
        volvol=np.array([1.0, 1.0], dtype=dtype),
        volvol_lower_bound=0.0,
        volvol_upper_bound=10.0,
        rho=np.array([0.0, 0.0], dtype=dtype),
        rho_lower_bound=-0.75,
        rho_upper_bound=0.75,
        maximum_iterations=1000)

    [calibrated_alpha, calibrated_beta, calibrated_volvol, calibrated_rho,
     is_converged] = self.evaluate(
         [models.alpha, models.beta, models.volvol, models.rho, is_converged])

    with self.subTest('AllConverged'):
      self.assertTrue(all(is_converged))
    with self.subTest('AlphaRecovered'):
      self.assertAllClose(calibrated_alpha, [1.5, 2.5], atol=2e-3, rtol=2e-3)
    with self.subTest('BetaRecovered'):
      self.assertAllClose(calibrated_beta, [0.5, 0.5], atol=2e-3, rtol=2e-3)
    with self.subTest('VolVolRecovered'):
      self.assertAllClose(calibrated_volvol, [0.33, 0.66], atol=2e-2, rtol=5e-2)
    with self.subTest('RhoRecovered'):
      self.assertAllClose(calibrated_rho, [0.1, -0.1], atol=1e-2, rtol=5e-2)

  def test_dynamic_shapes(self):
    """Test calibration function with dynamically shaped inputs."""
    dtype = np.float64

    observed_prices = np.array(
        [[20.09689284, 10.91953054, 4.25012702, 1.11561839, 0.20815853],
         [3.34813209, 6.03578711, 10.2874194, 16.26824328, 23.73850935]],
        dtype=dtype)

    strikes = np.array(
        [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=dtype)

    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    forwards = [[100.0], [100.0]]
    is_call_options = np.array([[True], [False]])

    beta = np.array([0.5], dtype=dtype)
    volvol = np.array([1.0], dtype=dtype)

    @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=dtype),
                                  tf.TensorSpec([None, None], dtype=dtype),
                                  tf.TensorSpec([None, None], dtype=dtype),
                                  tf.TensorSpec([None, None], dtype=dtype),
                                  tf.TensorSpec([None, None], dtype=tf.bool),
                                  tf.TensorSpec([None], dtype=dtype),
                                  tf.TensorSpec([None], dtype=dtype)])
    def fn(observed_prices, strikes, expiries, forwards, is_call_options,
           beta, volvol):
      models, is_converged, _ = tff.models.sabr.calibration(
          prices=observed_prices,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call_options,
          beta=beta,
          calibrate_beta=False,
          volvol=volvol,
          volvol_lower_bound=0.0,
          volvol_upper_bound=10.0,
          rho=np.array(0.0, dtype=dtype),
          rho_lower_bound=-0.75,
          rho_upper_bound=0.75,
          optimizer_fn=tff.math.optimizer.bfgs_minimize,
          maximum_iterations=1000)
      return models, is_converged

    models, is_converged = fn(observed_prices, strikes, expiries, forwards,
                              is_call_options, beta, volvol)
    [calibrated_alpha, calibrated_beta, calibrated_volvol, calibrated_rho,
     is_converged] = self.evaluate(
         [models.alpha, models.beta, models.volvol, models.rho, is_converged])
    with self.subTest('AllConverged'):
      self.assertTrue(all(is_converged))
    with self.subTest('AlphaRecovered'):
      self.assertAllClose(calibrated_alpha, [1.5, 2.5], atol=2e-3, rtol=2e-3)
    with self.subTest('BetaRecovered'):
      self.assertAllClose(calibrated_beta, [0.5, 0.5], atol=2e-3, rtol=2e-3)
    with self.subTest('VolVolRecovered'):
      self.assertAllClose(calibrated_volvol, [0.33, 0.66], atol=2e-2, rtol=5e-2)
    with self.subTest('RhoRecovered'):
      self.assertAllClose(calibrated_rho, [0.1, -0.1], atol=1e-2, rtol=5e-2)

  def test_calibration_batch_limits(self):
    """Demonstrate that lower/upper limits can be set independently in batch."""
    dtype = np.float64

    observed_prices = np.array(
        [[20.09689284, 10.91953054, 4.25012702, 1.11561839, 0.20815853],
         [3.34813209, 6.03578711, 10.2874194, 16.26824328, 23.73850935]],
        dtype=dtype)

    strikes = np.array(
        [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=dtype)
    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    forwards = 100.0
    is_call_options = np.array([[True], [False]])

    beta = np.array([0.5, 0.5], dtype=dtype)

    models, is_converged, _ = tff.models.sabr.calibration(
        prices=observed_prices,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        beta=beta,
        calibrate_beta=False,
        volvol=np.array([1.0, 1.0], dtype=dtype),
        volvol_lower_bound=np.array([0.0, 0.2]),
        volvol_upper_bound=np.array([5.0, 10.0]),
        rho=np.array([0.0, 0.0], dtype=dtype),
        rho_lower_bound=np.array([-1.0, -0.75]),
        rho_upper_bound=np.array([1.0, 0.75]),
        maximum_iterations=1000)

    (calibrated_alpha, calibrated_beta, calibrated_volvol, calibrated_rho,
     is_converged) = self.evaluate(
         [models.alpha, models.beta, models.volvol, models.rho, is_converged])

    self.assertTrue(all(is_converged))

    self.assertAllClose(calibrated_alpha, [1.5, 2.5], atol=2e-3, rtol=2e-3)
    self.assertAllClose(calibrated_beta, [0.5, 0.5], atol=2e-3, rtol=2e-3)
    self.assertAllClose(calibrated_volvol, [0.33, 0.66], atol=1e-2, rtol=5e-2)
    self.assertAllClose(calibrated_rho, [0.1, -0.1], atol=1e-2, rtol=5e-2)

  def test_validate_args(self):
    dtype = np.float64

    observed_prices = np.array(
        [[20.09689284, 10.91953054, 4.25012702, 1.11561839, 0.20815853],
         [3.34813209, 6.03578711, 10.2874194, 16.26824328, 23.73850935]],
        dtype=dtype)

    strikes = np.array(
        [[80.0, 90.0, 100.0, 110.0, 120.0], [80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=dtype)
    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    forwards = 100.0
    is_call_options = np.array([[True], [False]])

    beta = np.array([0.5, 0.5], dtype=dtype)

    # Fails because `volvol` is outside the limits.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      _, is_converged, _ = tff.models.sabr.calibration(
          prices=observed_prices,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call_options,
          beta=beta,
          calibrate_beta=False,
          volvol=np.array([1.0, 12.0], dtype=dtype),
          volvol_lower_bound=0.0,
          volvol_upper_bound=10.0,
          rho=np.array([0.0, 0.0], dtype=dtype),
          rho_lower_bound=-0.75,
          rho_upper_bound=0.75,
          validate_args=True)
      self.evaluate(is_converged)

    # Fails because `rho` exceeds its expected bounds.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      _, is_converged, _ = tff.models.sabr.calibration(
          prices=observed_prices,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call_options,
          beta=beta,
          calibrate_beta=False,
          volvol=np.array([1.0, 1.0], dtype=dtype),
          volvol_lower_bound=0.0,
          volvol_upper_bound=10.0,
          rho=np.array([0.0, 0.76], dtype=dtype),
          rho_lower_bound=-0.75,
          rho_upper_bound=0.75,
          validate_args=True)
      self.evaluate(is_converged)

    # Arguments are okay.
    _, is_converged, _ = tff.models.sabr.calibration(
        prices=observed_prices,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        beta=beta,
        calibrate_beta=False,
        volvol=np.array([1.0, 1.0], dtype=dtype),
        volvol_lower_bound=0.0,
        volvol_upper_bound=10.0,
        rho=np.array([0.0, 0.0], dtype=dtype),
        rho_lower_bound=-0.75,
        rho_upper_bound=0.75,
        validate_args=True)
    is_converged = self.evaluate(is_converged)
    self.assertTrue(all(is_converged))

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_noise_lognormal_fixed_beta_0x5_price_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': False,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': LOGNORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_lognormal_fixed_beta_0x5_vol_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': True,
          'max_iterations': 1000,
          'tolerance': 1e-8,
          'vol_type': LOGNORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_lognormal_fixed_beta_extremes_price_based',
          'true_alpha': np.array([10.0, 0.1], dtype=np.float64),
          'true_beta': np.array([0.0, 1.0], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': False,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': LOGNORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_lognormal_fixed_beta_extremes_vol_based',
          'true_alpha': np.array([10.0, 0.1], dtype=np.float64),
          'true_beta': np.array([0.0, 1.0], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': True,
          'max_iterations': 1000,
          'tolerance': 1e-8,
          'vol_type': LOGNORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'low_noise_lognormal_fixed_beta_0x5_price_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': False,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': LOGNORMAL,
          'noise_size': 0.01,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'low_noise_lognormal_fixed_beta_0x5_vol_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': True,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': LOGNORMAL,
          'noise_size': 0.01,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_lognormal_calib_beta_price_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.4, 0.6], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': True,
          'vol_based_calibration': False,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': LOGNORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_lognormal_calib_beta_vol_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.4, 0.6], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': True,
          'vol_based_calibration': True,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': LOGNORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_normal_fixed_beta_0x5_price_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': False,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': NORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_normal_fixed_beta_0x5_vol_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': True,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': NORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_normal_fixed_beta_extremes_price_based',
          'true_alpha': np.array([10.0, 0.1], dtype=np.float64),
          'true_beta': np.array([0.0, 1.0], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': False,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': NORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_normal_fixed_beta_extremes_vol_based',
          'true_alpha': np.array([10.0, 0.1], dtype=np.float64),
          'true_beta': np.array([0.0, 1.0], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': True,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': NORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'low_noise_normal_fixed_beta_0x5_price_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': False,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': NORMAL,
          'noise_size': 0.01,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'low_noise_normal_fixed_beta_0x5_vol_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.5, 0.5], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': False,
          'vol_based_calibration': True,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': NORMAL,
          'noise_size': 0.01,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_normal_calib_beta_price_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.4, 0.6], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': True,
          'vol_based_calibration': False,
          'max_iterations': 1000,
          'tolerance': 1e-5,
          'vol_type': NORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
      {
          'testcase_name': 'no_noise_normal_calib_beta_vol_based',
          'true_alpha': np.array([1.5, 2.5], dtype=np.float64),
          'true_beta': np.array([0.4, 0.6], dtype=np.float64),
          'true_volvol': np.array([0.33, 0.50], dtype=np.float64),
          'true_rho': np.array([0.1, -0.1], dtype=np.float64),
          'calibrate_beta': True,
          'vol_based_calibration': True,
          'max_iterations': 1000,
          'tolerance': 1e-6,
          'vol_type': NORMAL,
          'noise_size': 0.0,
          'price_tol': (1e-2, 5e-3)
      },
  )
  def test_calibration(self, true_alpha, true_beta, true_volvol, true_rho,
                       calibrate_beta, vol_based_calibration, max_iterations,
                       tolerance, vol_type, noise_size, price_tol):
    dtype = np.float64

    # Construct some market conditions.
    strikes = np.array([np.arange(95, 105.10, 0.1),
                        np.arange(95, 105.10, 0.1)],
                       dtype=dtype)
    expiries = np.array([[0.5], [1.0]], dtype=dtype)
    is_call_options = np.array([[True], [False]])
    forwards = 100.0

    # Generate some prices.
    denoised_prices = tff.models.sabr.approximations.european_option_price(
        forwards=forwards,
        strikes=strikes,
        expiries=expiries,
        is_call_options=is_call_options,
        alpha=np.expand_dims(true_alpha, axis=-1),
        beta=np.expand_dims(true_beta, axis=-1),
        volvol=np.expand_dims(true_volvol, axis=-1),
        rho=np.expand_dims(true_rho, axis=-1),
        volatility_type=vol_type,
        dtype=dtype)

    # Add noise to the prices
    observed_prices = denoised_prices + tf.random.stateless_normal(
        denoised_prices.shape,
        stddev=noise_size * denoised_prices,
        seed=[2, 4],
        dtype=dtype)

    # Calibrate the models.
    initial_beta = np.array([0.5, 0.5],
                            dtype=dtype) if calibrate_beta else true_beta
    models, _, _ = tff.models.sabr.calibration(
        prices=observed_prices,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        is_call_options=is_call_options,
        beta=initial_beta,
        calibrate_beta=calibrate_beta,
        volatility_based_calibration=vol_based_calibration,
        volvol=np.array([1.0, 1.0], dtype=dtype),
        volvol_lower_bound=0.0,
        volvol_upper_bound=10.0,
        rho=np.array([0.0, 0.0], dtype=dtype),
        rho_lower_bound=-0.75,
        rho_upper_bound=0.75,
        volatility_type=vol_type,
        maximum_iterations=max_iterations,
        tolerance=tolerance,
        dtype=dtype)

    (calibrated_alpha, calibrated_beta, calibrated_volvol,
     calibrated_rho) = self.evaluate(
         [models.alpha, models.beta, models.volvol, models.rho])

    # Back out the approximated prices from the calibrated model, and check that
    # they agree with our input prices (up to noise)
    calibrated_prices = tff.models.sabr.approximations.european_option_price(
        forwards=forwards,
        strikes=strikes,
        expiries=expiries,
        is_call_options=is_call_options,
        alpha=np.array(np.expand_dims(calibrated_alpha, axis=1), dtype=dtype),
        beta=np.array(np.expand_dims(calibrated_beta, axis=1), dtype=dtype),
        volvol=np.array(np.expand_dims(calibrated_volvol, axis=1), dtype=dtype),
        rho=np.array(np.expand_dims(calibrated_rho, axis=1), dtype=dtype),
        volatility_type=vol_type,
        dtype=dtype)

    calibrated_prices, denoised_prices = self.evaluate(
        [calibrated_prices, denoised_prices])
    self.assertAllClose(
        calibrated_prices,
        denoised_prices,
        atol=price_tol[0],
        rtol=price_tol[1])

if __name__ == '__main__':
  tf.test.main()
