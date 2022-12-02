from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


option_price_binomial = tff.black_scholes.option_price_binomial
option_price = tff.black_scholes.option_price
andersen_lake = tff.experimental.american_option_pricing.andersen_lake.andersen_lake


@test_util.run_all_in_graph_and_eager_modes
class ExerciseBoundaryTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'BatchRank1_Simple',
          'k': [0.6, 0.2, 5.0, 1.2, 2.2, 5.2],
          'tau': [1.0, 1.0, 2.0, 1.5, 1.5, 2.0],
          'r': [0.0, 0.01, 1.0, 0.03, 0.02, 0.01],
          'f': None,
          'q': [0.02, 0.07, 0.0, 0.01, 0.01, 0.0],
          'is_call_options': [True, False, True, True, False, False],
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': [2.0, 3.0, 5.0, 1.5, 2.5, 5.5],
          'forwards': None,
      },
      {
          'testcase_name': 'BatchRank1_CallSingleValue',
          'k': [2.0, 0.7, 5.0, 1.2, 2.2, 5.2],
          'tau': [1.0, 1.0, 0.5, 1.5, 1.5, 2.0],
          'r': [0.0, 0.01, 0.8, 0.03, 0.02, 0.01],
          'f': None,
          'q': [0.02, 0.07, 0.0, 0.01, 0.01, 0.0],
          'is_call_options': False,
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': [2.0, 1.0, 4.0, 1.5, 2.5, 5.5],
          'forwards': None,
      },
      {
          'testcase_name': 'BatchRank1_RNone',
          'k': [2.0, 0.3, 0.05, 1.2, 2.2, 5.2],
          'tau': [1.0, 1.0, 2.0, 1.5, 1.5, 2.0],
          'r': None,
          'f': [0.85, 0.75, 0.94, 0.93, 0.92, 0.91],
          'q': [0.02, 0.07, 0.0, 0.01, 0.01, 0.0],
          'is_call_options': [True, False, False, False, True, False],
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': [2.0, 3.0, 5.0, 1.5, 2.5, 5.5],
          'forwards': None,
      },
      {
          'testcase_name': 'BatchRank1_FAndRNone',
          'k': [2.0, 3.0, 5.0, 1.2, 0.2, 5.2],
          'tau': [1.0, 1.0, 2.0, 1.5, 1.5, 2.0],
          'r': None,
          'f': None,
          'q': [0.02, 0.07, 0.0, 0.01, 0.01, 0.0],
          'is_call_options': [True, False, False, False, True, False],
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': [2.0, 3.0, 5.0, 1.5, 2.5, 5.5],
          'forwards': None,
      },
      {
          'testcase_name': 'BatchRank1_QNone',
          'k': [2.0, 3.0, 5.0, 1.2, 2.2, 0.2],
          'tau': [1.0, 1.0, 2.0, 1.5, 1.5, 2.0],
          'r': [0.0, 0.1, 0.0, 0.03, 0.02, 0.01],
          'f': None,
          'q': None,
          'is_call_options': [True, False, False, False, True, False],
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': [2.0, 3.0, 5.0, 1.5, 2.5, 5.5],
          'forwards': None,
      },
      {
          'testcase_name': 'BatchRank1_SNone',
          'k': [2.0, 3.0, 5.0, 0.7, 2.2, 5.2],
          'tau': [1.0, 1.0, 2.0, 1.5, 1.5, 2.0],
          'r': [0.0, 0.1, 0.0, 0.03, 0.02, 0.01],
          'f': None,
          'q': [0.02, 0.07, 0.4, 0.01, 0.01, 0.0],
          'is_call_options': [True, False, False, False, True, False],
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': None,
          'forwards': [2.0, 3.0, 5.0, 1.5, 2.5, 5.5],
      },
      {
          'testcase_name': 'BatchRank1_RAndSNone',
          'k': [2.0, 3.0, 5.0, 0.7, 2.2, 5.2],
          'tau': [1.0, 1.0, 2.0, 1.5, 1.5, 2.0],
          'r': None,
          'f': [0.85, 0.75, 0.94, 0.93, 0.92, 0.91],
          'q': [0.02, 0.07, 0.4, 0.01, 0.01, 0.0],
          'is_call_options': [True, False, False, False, True, False],
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': None,
          'forwards': [2.0, 3.0, 5.0, 1.5, 2.5, 5.5],
      },
      {
          'testcase_name': 'BatchRank1_CallNone',
          'k': [2.0, 3.0, 5.0, 1.2, 0.2, 5.2],
          'tau': [1.0, 1.0, 2.0, 1.5, 1.5, 2.0],
          'r': [0.0, 0.1, 0.3, 0.03, 0.02, 0.01],
          'f': None,
          'q': [0.02, 0.07, 0.8, 0.01, 0.01, 0.0],
          'is_call_options': None,
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': [2.0, 3.0, 5.0, 1.5, 2.5, 5.5],
          'forwards': None,
      },
      )
  def test_andersen_lake_values(self, k, tau, r, f, q, is_call_options, sigma,
                                s, forwards, grid_num_points=30,
                                integration_num_points_kronrod=31,
                                integration_num_points_legendre=32,
                                tolerance_exercise_boundary=1e-11,
                                tolerance_kronrod=1e-11,
                                max_iterations_exercise_boundary=200,
                                max_depth_kronrod=50,
                                test_tolerance=1e-3,
                                dtype=tf.float64):
    tau_binomial = tf.constant(tau, dtype=dtype)
    r_binomial = r
    if f is not None:
      f = tf.constant(f, dtype=dtype)
      r_binomial = tf.math.divide_no_nan(-tf.math.log(f), tau_binomial)
    if q is not None:
      q_binomial = tf.constant(q, dtype=tf.float64)
    else:
      q_binomial = tf.constant([0.0], dtype=dtype)
    s_binomial = s
    if forwards is not None:
      forwards_binomial = tf.constant(forwards, dtype=dtype)
      s_binomial = forwards_binomial * tf.exp(
          -(r_binomial - q_binomial) * tau_binomial)
    expected = option_price_binomial(
        volatilities=sigma,
        strikes=k,
        expiries=tau,
        spots=s_binomial,
        discount_rates=r_binomial,
        dividend_rates=q,
        is_call_options=is_call_options,
        is_american=True,
        num_steps=1000,
        dtype=dtype)
    expected_prices = np.array(self.evaluate(expected))
    computed_prices = andersen_lake(
        volatilities=sigma,
        strikes=k,
        expiries=tau,
        spots=s,
        forwards=forwards,
        discount_rates=r,
        discount_factors=f,
        dividend_rates=q,
        is_call_options=is_call_options,
        grid_num_points=grid_num_points,
        integration_num_points_kronrod=integration_num_points_kronrod,
        integration_num_points_legendre=integration_num_points_legendre,
        max_iterations_exercise_boundary=max_iterations_exercise_boundary,
        max_depth_kronrod=max_depth_kronrod,
        tolerance_exercise_boundary=tolerance_exercise_boundary,
        tolerance_kronrod=tolerance_kronrod,
        dtype=dtype)

    computed_prices = np.array(self.evaluate(computed_prices))
    np.testing.assert_allclose(computed_prices, expected_prices,
                               atol=test_tolerance)

  @parameterized.named_parameters({
      'testcase_name': 'QZero',
      'k': [2.0, 4.0],
      'tau': [1.0, 0.4],
      'r': [0.0, 0.0],
      'q': [0.0, 0.0],
      'is_call_options': [False, True],
      'sigma': [0.2, 0.3],
      's': [2.0, 0.8],
      'grid_num_points': 40,
      'integration_num_points_kronrod': 31,
      'integration_num_points_legendre': 32,
      'tolerance_exercise_boundary': 1e-11,
      'tolerance_kronrod': 1e-11,
      'max_iterations_exercise_boundary': 400,
      'max_depth_kronrod': 50
  })
  def test_andersen_lake_european(
      self, k, tau, r, q, is_call_options, sigma, s, grid_num_points,
      integration_num_points_kronrod, integration_num_points_legendre,
      tolerance_exercise_boundary, tolerance_kronrod,
      max_iterations_exercise_boundary, max_depth_kronrod, dtype=tf.float64):
    # When r == q == 0 American option prices are the same as European.
    expected = option_price(volatilities=sigma, strikes=k,
                            expiries=tau, spots=s,
                            discount_rates=r, dividend_rates=q,
                            is_call_options=is_call_options,
                            dtype=dtype)
    expected_prices = np.array(self.evaluate(expected))
    computed_prices = andersen_lake(
        volatilities=sigma,
        strikes=k,
        expiries=tau,
        spots=s,
        discount_rates=r,
        dividend_rates=q,
        is_call_options=is_call_options,
        grid_num_points=grid_num_points,
        integration_num_points_kronrod=integration_num_points_kronrod,
        integration_num_points_legendre=integration_num_points_legendre,
        max_iterations_exercise_boundary=max_iterations_exercise_boundary,
        max_depth_kronrod=max_depth_kronrod,
        tolerance_exercise_boundary=tolerance_exercise_boundary,
        tolerance_kronrod=tolerance_kronrod,
        dtype=dtype)
    computed_prices = np.array(self.evaluate(computed_prices))
    np.testing.assert_allclose(computed_prices, expected_prices, rtol=1e-8)

  @parameterized.named_parameters(
      {
          'testcase_name': 'BatchRank1_TauZero',
          'k': [2.0, 3.0, 5.0, 1.2, 2.2, 5.2],
          'tau': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          'r': [0.0, 0.1, 0.0, 0.03, 0.02, 0.01],
          'f': None,
          'q': [0.02, 0.07, 0.0, 0.01, 0.01, 0.0],
          'is_call_options': [True, False, False, False, True, False],
          'sigma': [0.2, 0.3, 0.4, 0.15, 0.25, 0.35],
          's': [2.0, 3.0, 5.0, 1.5, 2.5, 5.5],
          'forwards': None,
          'grid_num_points': 30,
          'integration_num_points_kronrod': 31,
          'integration_num_points_legendre': 32,
          'tolerance_exercise_boundary': 1e-11,
          'tolerance_kronrod': 1e-11,
          'max_iterations_exercise_boundary': 200,
          'max_depth_kronrod': 50,
          'test_tolerance': 1e-16,
          'dtype': tf.float64,
      },
  )
  def test_andersen_lake_tau_zero(self, k, tau, r, f, q, is_call_options, sigma,
                                  s, forwards, grid_num_points,
                                  integration_num_points_kronrod,
                                  integration_num_points_legendre,
                                  tolerance_exercise_boundary,
                                  tolerance_kronrod,
                                  max_iterations_exercise_boundary,
                                  max_depth_kronrod, test_tolerance, dtype):
    k_tensor = tf.constant(k, dtype=dtype)
    s_tensor = tf.constant(s, dtype=dtype)
    is_call_options_tensor = tf.constant(is_call_options, dtype=tf.bool)
    expected = tf.where(is_call_options_tensor,
                        tf.math.maximum(s_tensor - k_tensor, 0),
                        tf.math.maximum(k_tensor - s_tensor, 0))
    expected_prices = np.array(self.evaluate(expected))
    computed_prices = andersen_lake(
        volatilities=sigma,
        strikes=k,
        expiries=tau,
        spots=s,
        forwards=forwards,
        discount_rates=r,
        discount_factors=f,
        dividend_rates=q,
        is_call_options=is_call_options,
        grid_num_points=grid_num_points,
        integration_num_points_kronrod=integration_num_points_kronrod,
        integration_num_points_legendre=integration_num_points_legendre,
        max_iterations_exercise_boundary=max_iterations_exercise_boundary,
        max_depth_kronrod=max_depth_kronrod,
        tolerance_exercise_boundary=tolerance_exercise_boundary,
        tolerance_kronrod=tolerance_kronrod,
        dtype=dtype)
    computed_prices = np.array(self.evaluate(computed_prices))
    np.testing.assert_allclose(computed_prices, expected_prices,
                               rtol=test_tolerance)

if __name__ == '__main__':
  tf.test.main()
