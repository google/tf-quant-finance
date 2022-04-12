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
"""Tests for Sabr Approximations of Implied Volatilities."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

# Helper aliases.
NORMAL = tff.models.sabr.approximations.SabrImpliedVolatilityType.NORMAL
LOGNORMAL = tff.models.sabr.approximations.SabrImpliedVolatilityType.LOGNORMAL


@test_util.run_all_in_graph_and_eager_modes
class SabrApproximationImpliedVolatilityTest(parameterized.TestCase,
                                             tf.test.TestCase):
  """Tests for the SABR approximations to implied volatility."""

  def test_implied_volatility_docstring_example(self):
    equiv_vol = tff.models.sabr.approximations.implied_volatility(
        forwards=np.array([120.0, 20.0]),
        strikes=np.array([106.0, 11.0]),
        expiries=np.array([17.0 / 365.0, 400.0 / 365.0]),
        alpha=1.63,
        beta=0.6,
        rho=0.00002,
        volvol=3.3,
        dtype=tf.float64)
    equiv_vol = self.evaluate(equiv_vol)

    # Answer obtained by the equivalent formulation in QuantLib, e.g.:
    #
    # ```
    # import QuantLib as ql
    #
    # print(
    #   ql.sabrVolatility(
    #     strike=106.0, forward=120.0, expiryTime=17.0/365.0,
    #     alpha=1.63, beta=0.6, volvol=3.3, rho=0.0002
    #   ))
    # ```
    self.assertAllClose(equiv_vol, [0.33284656705268817, 1.9828728139982792])

  def test_implied_volatility_parameter_broadcasting(self):
    equiv_vol = tff.models.sabr.approximations.implied_volatility(
        strikes=np.array([
            [105.0, 110.0, 115.0, 120.0, 125.0],
            [105.0, 110.0, 115.0, 120.0, 125.0],
            [105.0, 110.0, 115.0, 120.0, 125.0],
        ]),
        expiries=np.array([
            [30.0 / 365.0],
            [60.0 / 365.0],
            [90.0 / 365.0],
        ]),
        forwards=115.0,
        alpha=np.array([
            [1.00],
            [1.50],
            [2.00],
        ]),
        beta=np.array([
            [0.25],
            [0.50],
            [0.75],
        ]),
        rho=0.05,
        volvol=np.array([
            [2.0],
            [2.5],
            [3.0],
        ]),
        dtype=tf.float64)
    equiv_vol = self.evaluate(equiv_vol)

    self.assertAllClose(
        equiv_vol,
        [
            [
                # These correspond to:
                # strikes=[105.0, 110.0, 115.0, 120.0, 125.0],
                # For expiries=30.0/365.0, forwards=115.0, alpha=1.0, beta=0.25,
                # rho=0.05, and volvol=2.0.
                0.07290126718857293,
                0.048678694833488044,
                0.02925354052258421,
                0.048663433935318724,
                0.06968554457600247,
            ],
            [
                # These correspond to:
                # strikes=[105.0, 110.0, 115.0, 120.0, 125.0],
                # For expiries=60.0/365.0, forwards=115.0, alpha=1.5, beta=0.5,
                # rho=0.05, and volvol=2.5.
                0.19479147227269086,
                0.1647194927224411,
                0.15186141282688834,
                0.16548343518393138,
                0.19098676373872667,
            ],
            [
                # These correspond to:
                # strikes=[105.0, 110.0, 115.0, 120.0, 125.0],
                # For expiries=90.0/365.0, forwards=115.0, alpha=2.0, beta=0.75,
                # rho=0.05, and volvol=3.0.
                0.7493941577599591,
                0.7318045979333434,
                0.7259931664225101,
                0.7310817574988578,
                0.7449168095317619,
            ],
        ])

  @parameterized.product(
      (
          # 'testcase_name': 'generic_example',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.5,
              'volvol': 2.8,
              'rho': -0.02,
              'expected_vol': 0.1805437461785543
          },
          # 'testcase_name': 'beta_zero',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.0,
              'volvol': 2.8,
              'rho': 0.2,
              'expected_vol': 0.10940855660611389
          },
          # 'testcase_name': 'beta_one',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 1.0,
              'volvol': 2.8,
              'rho': 0.2,
              'expected_vol': 0.5679836366288498
          },
          # 'testcase_name': 'volvol_zero',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.55,
              'volvol': 0.0,
              'rho': 0.2,
              'expected_vol': 0.036000740359707344
          },
          # 'testcase_name': 't_ex_zero',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 0.0,
              'alpha': 0.31,
              'beta': 0.55,
              'volvol': 2.5,
              'rho': 0.2,
              'expected_vol': 0.1400903903272142
          },
          # 'testcase_name': 'at_the_money',
          {
              'strikes': 130.0,
              'forwards': 130.0,
              'expiries': 0.5,
              'alpha': 0.31,
              'beta': 0.55,
              'volvol': 2.5,
              'rho': 0.2,
              'expected_vol': 0.043211700217443874
          },
          # 'testcase_name': 'tensor_inputs',
          {
              'strikes':
                  np.array([11.0, 15.0]),
              'forwards':
                  np.array([10.0, 11.0]),
              'expiries':
                  np.array([0.5, 2.0]),
              'alpha':
                  0.31,
              'beta':
                  0.28,
              'volvol':
                  2.5,
              'rho':
                  0.2,
              'expected_vol':
                  np.array([
                      0.14847284969337574,
                      0.46783426296790165,
                  ])
          }),
      dtype=(tf.float32, tf.float64))
  def test_implied_volatility_lognormal_correctness(self, forwards, strikes,
                                                    expiries, alpha, beta, rho,
                                                    volvol, expected_vol,
                                                    dtype):
    equiv_vol = tff.models.sabr.approximations.implied_volatility(
        forwards=forwards,
        strikes=strikes,
        expiries=expiries,
        alpha=alpha,
        beta=beta,
        rho=rho,
        volvol=volvol,
        dtype=dtype)
    equiv_vol = self.evaluate(equiv_vol)
    self.assertAllClose(expected_vol, equiv_vol, atol=2e-5, rtol=2e-4)

  @parameterized.product(
      (
          # 'testcase_name': 'generic_example',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.12,
              'volvol': 2.8,
              'rho': -0.02,
              'expected_vol': 14.240742
          },
          # 'testcase_name': 'beta_zero',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.0,
              'volvol': 2.8,
              'rho': 0.2,
              'expected_vol': 13.098578
          },
          # 'testcase_name': 'beta_one',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 1.0,
              'volvol': 2.8,
              'rho': 0.2,
              'expected_vol': 67.89029
          },
          # 'testcase_name': 'volvol_zero',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.55,
              'volvol': 0.0,
              'rho': 0.2,
              'expected_vol': 4.309942
          },
          # 'testcase_name': 't_ex_zero',
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 0.0,
              'alpha': 0.31,
              'beta': 0.55,
              'volvol': 2.5,
              'rho': 0.2,
              'expected_vol': 16.771861
          },
          # 'testcase_name': 'tensor_inputs',
          {
              'strikes': np.array([11.0, 15.0]),
              'forwards': np.array([10.0, 11.0]),
              'expiries': np.array([0.5, 2.0]),
              'alpha': 0.31,
              'beta': 0.28,
              'volvol': 2.5,
              'rho': 0.2,
              'expected_vol': np.array([1.557701, 6.032939])
          }),
      dtype=(tf.float32, tf.float64))
  def test_implied_volatility_normal_correctness(self, forwards, strikes,
                                                 expiries, alpha, beta, rho,
                                                 volvol,
                                                 expected_vol, dtype):
    equiv_vol = tff.models.sabr.approximations.implied_volatility(
        forwards=forwards,
        strikes=strikes,
        expiries=expiries,
        alpha=alpha,
        beta=beta,
        rho=rho,
        volvol=volvol,
        dtype=dtype,
        volatility_type=NORMAL)
    equiv_vol = self.evaluate(equiv_vol)
    self.assertAllClose(expected_vol, equiv_vol, atol=5e-3, rtol=1e-3)

  @parameterized.product(
      (
          # Generic scalar example.
          {
              'strikes': 130.0,
              'forwards': 110.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.5,
              'volvol': 2.8,
              'rho': -0.02,
          },
          # Generic scalar pathological example: at-the-money.
          {
              'strikes': 130.0,
              'forwards': 130.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.5,
              'volvol': 2.8,
              'rho': -0.02,
          },
          # Generic scalar pathological example: beta = 0
          {
              'strikes': 130.0,
              'forwards': 130.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.0,
              'volvol': 2.8,
              'rho': -0.02,
          },
          # Generic scalar pathological example: beta = 1
          {
              'strikes': 130.0,
              'forwards': 130.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 1.0,
              'volvol': 2.8,
              'rho': -0.02,
          },
          # Generic scalar pathological example: volvol = 0
          {
              'strikes': 130.0,
              'forwards': 130.0,
              'expiries': 200.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.5,
              'volvol': 0.0,
              'rho': -0.02,
          },
          # Generic scalar pathological example: expiries = 0.0
          {
              'strikes': 130.0,
              'forwards': 130.0,
              'expiries': 0.0 / 365.0,
              'alpha': 0.31,
              'beta': 0.5,
              'volvol': 2.8,
              'rho': -0.02,
          },
          # Generic example, with tensor values, including at-the-money case and
          # expiry = 0.0 case.
          {
              'strikes':
                  np.array([[130.0, 140.0, 150.0], [130.0, 140.0, 150.0]]),
              'forwards':
                  140.0,
              'expiries': [[0.0], [1.0]],
              'alpha': [[0.25], [0.5]],
              'beta': [[0.33], [0.66]],
              'volvol': [[1.0], [2.0]],
              'rho': [[0.001], [-0.001]],
          }),
      vol_type=(NORMAL, LOGNORMAL),
      dtype=(tf.float32, tf.float64))
  def test_implied_volatility_differentiable(self, strikes, forwards, expiries,
                                             alpha, beta, volvol, rho, vol_type,
                                             dtype):
    forwards = tf.convert_to_tensor(forwards, dtype=dtype)
    strikes = tf.convert_to_tensor(strikes, dtype=dtype)
    expiries = tf.convert_to_tensor(expiries, dtype=dtype)
    alpha = tf.convert_to_tensor(alpha, dtype=dtype)
    beta = tf.convert_to_tensor(beta, dtype=dtype)
    rho = tf.convert_to_tensor(rho, dtype=dtype)
    volvol = tf.convert_to_tensor(volvol, dtype=dtype)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([forwards, strikes, expiries, alpha, beta, rho, volvol])
      equiv_vol = tff.models.sabr.approximations.implied_volatility(
          forwards=forwards,
          strikes=strikes,
          expiries=expiries,
          alpha=alpha,
          beta=beta,
          rho=rho,
          volvol=volvol,
          volatility_type=vol_type,
          dtype=dtype)
      grad = tape.gradient(
          target=equiv_vol,
          sources=[forwards, strikes, expiries, alpha, beta, rho, volvol])

    grad = self.evaluate(grad)
    self.assertTrue(all(np.all(np.isfinite(x)) for x in grad))

  @parameterized.parameters([NORMAL, LOGNORMAL])
  def test_implied_volatility_shifted_model_example(self, volatility_type):
    shift = np.array([1.0, 2.0])

    actual_vol = tff.models.sabr.approximations.implied_volatility(
        forwards=np.array([-0.5, 0.0]),
        strikes=np.array([1.0, 2.0]),
        expiries=np.array([17.0 / 365.0, 400.0 / 365.0]),
        alpha=1.63,
        beta=0.6,
        rho=0.00002,
        volvol=3.3,
        shift=shift,
        dtype=tf.float64,
        volatility_type=volatility_type)

    expected_vol = tff.models.sabr.approximations.implied_volatility(
        forwards=np.array([-0.5 + shift[0], 0.0 + shift[1]]),
        strikes=np.array([1.0 + shift[0], 2.0 + shift[1]]),
        expiries=np.array([17.0 / 365.0, 400.0 / 365.0]),
        alpha=1.63,
        beta=0.6,
        rho=0.00002,
        volvol=3.3,
        dtype=tf.float64,
        volatility_type=volatility_type)

    actual_vol, expected_vol = self.evaluate([actual_vol, expected_vol])
    self.assertAllClose(actual_vol, expected_vol)


if __name__ == '__main__':
  tf.test.main()
