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
"""Tests for vanilla_price."""

import functools

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class VanillaPrice(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods for the vanilla pricing module."""

  def test_option_prices(self):
    """Tests that the BS prices are correct."""
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    expiries = 1.0
    computed_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards))
    expected_prices = np.array(
        [0.0, 2.0, 2.0480684764112578, 1.0002029716043364, 2.0730313058959933])
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_option_prices_normal(self):
    """Tests that the prices using normal model are correct."""
    forwards = np.array([0.01, 0.02, 0.03, 0.03, 0.05])
    strikes = np.array([0.03, 0.03, 0.03, 0.03, 0.03])
    volatilities = np.array([0.0001, 0.001, 0.01, 0.005, 0.02])
    expiries = 1.0
    computed_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_normal_volatility=True))

    expected_prices = np.array(
        [0.0, 0.0, 0.0039894228040143, 0.0019947114020072, 0.0216663094117537])
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_zero_vol(self):
    """Tests that zero volatility is handled correctly."""
    # If the volatility is zero, the option's value should be correct.
    forwards = np.array([1.0, 1.0, 1.0, 1.0])
    strikes = np.array([1.1, 0.9, 1.1, 0.9])
    volatilities = np.array([0.0, 0.0, 0.0, 0.0])
    expiries = 1.0
    is_call_options = np.array([True, True, False, False])
    expected_prices = np.array([0.0, 0.1, 0.1, 0.0])
    computed_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_zero_expiry(self):
    """Tests that zero expiry is correctly handled."""
    # If the expiry is zero, the option's value should be correct.
    forwards = np.array([1.0, 1.0, 1.0, 1.0])
    strikes = np.array([1.1, 0.9, 1.1, 0.9])
    volatilities = np.array([0.1, 0.2, 0.5, 0.9])
    expiries = 0.0
    is_call_options = np.array([True, True, False, False])
    expected_prices = np.array([0.0, 0.1, 0.1, 0.0])
    computed_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_long_expiry_calls(self):
    """Tests that very long expiry call option behaves like the asset."""
    forwards = np.array([1.0, 1.0, 1.0, 1.0])
    strikes = np.array([1.1, 0.9, 1.1, 0.9])
    volatilities = np.array([0.1, 0.2, 0.5, 0.9])
    expiries = 1e10
    expected_prices = forwards
    computed_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_long_expiry_puts(self):
    """Tests that very long expiry put option is worth the strike."""
    forwards = np.array([1.0, 1.0, 1.0, 1.0])
    strikes = np.array([0.1, 10.0, 3.0, 0.0001])
    volatilities = np.array([0.1, 0.2, 0.5, 0.9])
    expiries = 1e10
    expected_prices = strikes
    computed_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=False))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_vol_and_expiry_scaling(self):
    """Tests that the price is invariant under vol->k vol, T->T/k**2."""
    np.random.seed(1234)
    n = 20
    forwards = np.exp(np.random.randn(n))
    volatilities = np.exp(np.random.randn(n) / 2)
    strikes = np.exp(np.random.randn(n))
    expiries = np.exp(np.random.randn(n))
    scaling = 5.0
    base_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards))
    scaled_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities * scaling,
            strikes=strikes,
            expiries=expiries / scaling / scaling,
            forwards=forwards))
    self.assertArrayNear(base_prices, scaled_prices, 1e-10)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      },
      {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      },
  )
  def test_option_prices_detailed_discount(self, dtype):
    """Tests the prices with discount_rates."""
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    discount_rates = 0.08
    volatilities = 0.2
    expiries = 0.25

    is_call_options = np.array([True] * 5 + [False] * 5)
    dividend_rates = 0.12
    computed_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            spots=spots,
            discount_rates=discount_rates,
            dividend_rates=dividend_rates,
            is_call_options=is_call_options,
            dtype=dtype))
    expected_prices = np.array(
        [0.03, 0.57, 3.42, 9.85, 18.62, 20.41, 11.25, 4.40, 1.12, 0.18])
    self.assertArrayNear(expected_prices, computed_prices, 5e-3)

  def test_binary_prices(self):
    """Tests that the BS binary option prices are correct."""
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    expiries = 1.0
    computed_prices = self.evaluate(
        tff.black_scholes.binary_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards))
    expected_prices = np.array([0.0, 0.0, 0.15865525, 0.99764937, 0.85927418])
    self.assertArrayNear(expected_prices, computed_prices, 1e-8)

  def test_binary_prices_bulk(self):
    """Tests unit of cash binary option pricing over a wide range of settings.

    Uses the fact that if the underlying follows a geometric brownian motion
    then, given the mean on the exponential scale and the variance on the log
    scale, the mean on the log scale is known. In particular for underlying S
    with forward price F, strike K, volatility sig, and expiry T:

    log(S) ~ N(log(F) - sig^2 T, sig^2 T)

    The price of the binary call option is the discounted probability that S
    will be greater than K at expiry (and for a put option, less than K). Since
    quantiles are preserved under monotonic transformations we can find this
    probability on the log scale. This provides an alternate calculation for the
    same price which we can use to corroborate the standard method.
    """
    np.random.seed(321)
    num_examples = 1000
    forwards = np.exp(np.random.normal(size=num_examples))
    strikes = np.exp(np.random.normal(size=num_examples))
    volatilities = np.exp(np.random.normal(size=num_examples))
    expiries = np.random.gamma(shape=1.0, scale=1.0, size=num_examples)
    log_scale = np.sqrt(expiries) * volatilities
    log_loc = np.log(forwards) - 0.5 * log_scale**2
    call_options = np.random.binomial(n=1, p=0.5, size=num_examples)
    discount_factors = np.random.beta(a=1.0, b=1.0, size=num_examples)

    cdf_values = self.evaluate(
        tfp.distributions.Normal(loc=log_loc,
                                 scale=log_scale).cdf(np.log(strikes)))

    expected_prices = discount_factors * (
        call_options + ((-1.0)**call_options) * cdf_values)

    is_call_options = np.array(call_options, dtype=np.bool)
    computed_prices = self.evaluate(
        tff.black_scholes.binary_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
            discount_factors=discount_factors))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_binary_vanilla_call_consistency(self):
    r"""Tests code consistency through relationship of binary and vanilla prices.

    With forward F, strike K, discount rate r, and expiry T, a vanilla call
    option should have price CV:

    $$ VC(K) = e^{-rT}( N(d_1)F - N(d_2)K ) $$

    A unit of cash paying binary call option should have price BC:

    $$ BC(K) = e^{-rT} N(d_2) $$

    Where d_1 and d_2 are standard Black-Scholes quanitities and depend on K
    through the ratio F/K. Hence for a small increment e:

    $$ (VC(K + e) - Vc(K))/e \approx -N(d_2)e^{-rT} = -BC(K + e) $$

    Similarly, for a vanilla put:

    $$ (VP(K + e) - VP(K))/e \approx N(-d_2)e^{-rT} = BP(K + e) $$

    This enables a test for consistency of pricing between vanilla and binary
    options prices.
    """
    np.random.seed(135)
    num_examples = 1000
    forwards = np.exp(np.random.normal(size=num_examples))
    strikes_0 = np.exp(np.random.normal(size=num_examples))
    epsilon = 1e-8
    strikes_1 = strikes_0 + epsilon
    volatilities = np.exp(np.random.normal(size=num_examples))
    expiries = np.random.gamma(shape=1.0, scale=1.0, size=num_examples)
    call_options = np.random.binomial(n=1, p=0.5, size=num_examples)
    is_call_options = np.array(call_options, dtype=np.bool)
    discount_factors = np.ones_like(forwards)

    option_prices_0 = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes_0,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options))

    option_prices_1 = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes_1,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options))

    binary_approximation = (-1.0)**call_options * (option_prices_1 -
                                                   option_prices_0) / epsilon

    binary_prices = self.evaluate(
        tff.black_scholes.binary_price(
            volatilities=volatilities,
            strikes=strikes_1,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
            discount_factors=discount_factors))

    self.assertArrayNear(binary_approximation, binary_prices, 1e-6)

  def test_binary_vanilla_consistency_exact(self):
    """Tests that the binary price is the negative gradient of vanilla price."""

    # The binary call option payoff is 1 when spot > strike and 0 otherwise.
    # This payoff is the proportional to the gradient of the payoff of a vanilla
    # call option (max(S-K, 0)) with respect to K. This test verifies that this
    # relationship is satisfied. A similar relation holds true between vanilla
    # puts and binary puts.
    dtype = np.float64
    strikes = tf.constant([1.0, 2.0], dtype=dtype)
    spots = tf.constant([1.5, 1.5], dtype=dtype)
    expiries = tf.constant([2.1, 1.3], dtype=dtype)
    discount_rates = tf.constant([0.03, 0.04], dtype=dtype)
    discount_factors = tf.exp(-discount_rates * expiries)
    is_call_options = tf.constant([True, False])
    volatilities = tf.constant([0.3, 0.4], dtype=dtype)
    actual_binary_price = self.evaluate(
        tff.black_scholes.binary_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            spots=spots,
            discount_factors=discount_factors,
            is_call_options=is_call_options))
    price_fn = functools.partial(
        tff.black_scholes.option_price,
        volatilities=volatilities,
        spots=spots,
        expiries=expiries,
        discount_rates=discount_rates,
        is_call_options=is_call_options)
    implied_binary_price = tff.math.fwd_gradient(lambda x: price_fn(strikes=x),
                                                 strikes)
    implied_binary_price = self.evaluate(
        tf.where(is_call_options, -implied_binary_price, implied_binary_price))
    self.assertArrayNear(implied_binary_price, actual_binary_price, 1e-10)

  def test_asset_or_nothing_prices(self):
    """Tests that the BS asset-or-nothing option prices are correct."""
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    expiries = 1.0
    computed_prices = self.evaluate(
        tff.black_scholes.asset_or_nothing_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards))
    expected_prices = np.array([0., 2., 2.52403424, 3.99315108, 4.65085383])

    self.assertArrayNear(expected_prices, computed_prices, 1e-8)

    is_call_options = True
    vanilla_prices = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
        ))
    cash_or_nothing_prices = self.evaluate(strikes *
                                           tff.black_scholes.binary_price(
                                               volatilities=volatilities,
                                               strikes=strikes,
                                               expiries=expiries,
                                               forwards=forwards,
                                               is_call_options=is_call_options,
                                           ))
    asset_or_nothing_prices = self.evaluate(
        tff.black_scholes.asset_or_nothing_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
        ))

    self.assertArrayNear(vanilla_prices,
                         asset_or_nothing_prices - cash_or_nothing_prices,
                         1e-10)

  @parameterized.product(
      discount_mode=('rates', 'factors'),
      is_normal=(True, False),
  )
  def test_vanilla_and_binary_prices_consistency_bulk(self, discount_mode,
                                                      is_normal):
    """Tests the consistency between vanilla and binary option prices."""

    # A vanilla call is equivalent to the combination of a long asset-or-nothing
    # call and short {strike} units of cash-or-nothing calls.
    # A vanilla put is equivalent to the combination of a long {strike} units of
    # cash-or-nothing puts and an short asset-or-nothing put.
    # This test confirms that the computed vanilla and binary option prices are
    # consistent with the above relations for a range of simulated settings.

    np.random.seed(321)
    num_examples = 1000
    volatilities = np.exp(np.random.normal(size=num_examples))
    strikes = np.exp(np.random.normal(size=num_examples))
    expiries = np.random.gamma(shape=1.0, scale=1.0, size=num_examples)
    forwards = np.exp(np.random.normal(size=num_examples))
    if discount_mode == 'rates':
      discount_rates = np.random.uniform(0.0, 0.05, size=num_examples)
      discount_factors = None
    else:
      discount_factors = np.random.beta(a=1.0, b=1.0, size=num_examples)
      discount_rates = None
    dividend_rates = np.random.uniform(0.0, 0.05, size=num_examples)
    call_options = np.random.binomial(n=1, p=0.5, size=num_examples)
    is_call_options = np.array(call_options, dtype=np.bool)

    asset_or_nothing_prices = tff.black_scholes.asset_or_nothing_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        is_call_options=is_call_options,
        is_normal_volatility=is_normal,
        discount_factors=discount_factors)
    cash_or_nothing_prices = tff.black_scholes.binary_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        is_call_options=is_call_options,
        is_normal_volatility=is_normal,
        discount_factors=discount_factors)
    synthetic_vanilla_from_binary_prices = tf.where(
        is_call_options,
        asset_or_nothing_prices - strikes * cash_or_nothing_prices,
        strikes * cash_or_nothing_prices - asset_or_nothing_prices)

    directly_computed_vanilla_prices = tff.black_scholes.option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        is_call_options=is_call_options,
        is_normal_volatility=is_normal,
        discount_factors=discount_factors)

    self.assertArrayNear(
        self.evaluate(synthetic_vanilla_from_binary_prices),
        self.evaluate(directly_computed_vanilla_prices), 1e-10)

  @parameterized.product(
      (
          {
              'vol': 0.25,
              'strikes': 90.0,
              'expiries': 0.5,
              'forwards': 100.0
          },
          {
              'vol': 0.25,
              'strikes': 90.0,
              'expiries': 0.0,  # Differentiable even at 0.0 expiry.
              'forwards': 100.0
          },
          {
              'vol': 0.25,
              'strikes': 90.0,
              'expiries': 0.5,
              'forwards': 90.0  # Differentiable at the money.
          },
      ),
      is_call=(True, False),
      is_normal=(True, False),
  )
  def test_vanilla_options_price_gradient_continuous(self, vol, strikes,
                                                     expiries, forwards,
                                                     is_call, is_normal):
    """Tests that the gradient exists, and is also right-continuous."""
    dtype = tf.float64
    vol = tf.convert_to_tensor(vol, dtype=dtype)
    strikes = tf.convert_to_tensor(strikes, dtype=dtype)
    expiries = tf.convert_to_tensor(expiries, dtype=dtype)
    forwards = tf.convert_to_tensor(forwards, dtype=dtype)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([vol, strikes, expiries, forwards])
      price = tff.black_scholes.option_price(
          volatilities=vol,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call,
          is_normal_volatility=is_normal,
          dtype=dtype)
      grad = tape.gradient(
          target=price, sources=[vol, strikes, expiries, forwards])

    grad = self.evaluate(grad)
    self.assertTrue(all(np.all(np.isfinite(x)) for x in grad))

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([vol, strikes, expiries, forwards])
      price_perturb_vol = tff.black_scholes.option_price(
          volatilities=vol + 1e-6,
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call,
          is_normal_volatility=is_normal,
          dtype=dtype)
      grad_perturb_vol = tape.gradient(
          target=price_perturb_vol, sources=[vol, strikes, expiries, forwards])

    grad_perturb_vol = self.evaluate(grad_perturb_vol)
    self.assertAllClose(grad, grad_perturb_vol, rtol=1e-4)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([vol, strikes, expiries, forwards])
      price_perturb_expiries = tff.black_scholes.option_price(
          volatilities=vol,
          strikes=strikes,
          expiries=expiries + 1e-6,
          forwards=forwards,
          is_call_options=is_call,
          is_normal_volatility=is_normal,
          dtype=dtype)
      grad_perturb_expiries = tape.gradient(
          target=price_perturb_expiries,
          sources=[vol, strikes, expiries, forwards])

    grad_perturb_expiries = self.evaluate(grad_perturb_expiries)
    self.assertAllClose(grad, grad_perturb_expiries, rtol=1e-4)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([vol, strikes, expiries, forwards])
      price_perturb_strikes = tff.black_scholes.option_price(
          volatilities=vol,
          strikes=strikes + 1e-6,
          expiries=expiries,
          forwards=forwards,
          is_call_options=is_call,
          is_normal_volatility=is_normal,
          dtype=dtype)
      grad_perturb_strikes = tape.gradient(
          target=price_perturb_strikes,
          sources=[vol, strikes, expiries, forwards])

    grad_perturb_strikes = self.evaluate(grad_perturb_strikes)
    self.assertAllClose(grad, grad_perturb_strikes, rtol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ScalarInputsUIP',
          'volatilities': 0.25,
          'strikes': 90.0,
          'expiries': 0.5,
          'spots': 100.0,
          'discount_rates': 0.08,
          'dividend_rates': 0.04,
          'barriers': 105.0,
          'rebates': 3.0,
          'is_barrier_down': False,
          'is_knock_out': False,
          'is_call_options': False,
          'expected_price': 1.4653,
      }, {
          'testcase_name': 'ScalarInputsUIP_Default_Values',
          'volatilities': 0.30,
          'strikes': 105.0,
          'expiries': 10.0,
          'spots': 100.0,
          'discount_rates': None,
          'dividend_rates': None,
          'barriers': 90.0,
          'rebates': None,
          'is_barrier_down': True,
          'is_knock_out': True,
          'is_call_options': True,
          'expected_price': 9.2848,
      }, {
          'testcase_name':
              'VectorInputs',
          'volatilities': [.25, .25, .25, .25, .25, .25, .25, .25],
          'strikes': [90., 90., 90., 90., 90., 90., 90., 90.],
          'expiries': [.5, .5, .5, .5, .5, .5, .5, .5],
          'spots': [100., 100., 100., 100., 100., 100., 100., 100.],
          'discount_rates': [.08, .08, .08, .08, .08, .08, .08, .08],
          'dividend_rates': [.04, .04, .04, .04, .04, .04, .04, .04],
          'barriers': [95., 95., 105., 105., 95., 105., 95., 105.],
          'rebates': [3., 3., 3., 3., 3., 3., 3., 3.],
          'is_barrier_down':
              [True, True, False, False, True, False, True, False],
          'is_knock_out': [True, False, True, False, True, True, False, False],
          'is_call_options':
              [True, True, True, True, False, False, False, False],
          'expected_price':
              [9.024, 7.7627, 2.6789, 14.1112, 2.2798, 3.7760, 2.95586, 1.4653],
      }, {
          'testcase_name':
              'MatrixInputs',
          'volatilities': [[.25, .25], [.25, .25], [.25, .25], [.25, .25]],
          'strikes': [[90., 90.], [90., 90.], [90., 90.], [90., 90.]],
          'expiries': [[.5, .5], [.5, .5], [.5, .5], [.5, .5]],
          'spots': [[100., 100.], [100., 100.], [100., 100.], [100., 100.]],
          'discount_rates': [[.08, .08], [.08, .08], [.08, .08], [.08, .08]],
          'dividend_rates': [[.04, .04], [.04, .04], [.04, .04], [.04, .04]],
          'barriers': [[95., 95.], [105., 105.], [95., 105.], [95., 105.]],
          'rebates': [[3., 3.], [3., 3.], [3., 3.], [3., 3.]],
          'is_barrier_down': [[True, True], [False, False], [True, False],
                              [True, False]],
          'is_knock_out': [[True, False], [True, False], [True, True],
                           [False, False]],
          'is_call_options': [[True, True], [True, True], [False, False],
                              [False, False]],
          'expected_price': [[9.024, 7.7627], [2.6789, 14.1112],
                             [2.2798, 3.7760], [2.95586, 1.4653]],
      }, {
          'testcase_name': 'dividend_rates',
          'volatilities': 0.25,
          'strikes': 90.0,
          'expiries': 0.5,
          'spots': 100.0,
          'discount_rates': 0.08,
          'dividend_rates': 0.04,
          'barriers': 105.0,
          'rebates': 3.0,
          'is_barrier_down': False,
          'is_knock_out': False,
          'is_call_options': False,
          'expected_price': 1.4653,
      })
  def test_barrier_option(self,
                          *,
                          volatilities,
                          strikes,
                          expiries,
                          spots,
                          discount_rates,
                          barriers,
                          rebates,
                          is_barrier_down,
                          is_knock_out,
                          is_call_options,
                          expected_price,
                          dividend_rates=None):
    """Computes test barrier option prices for the parameterized inputs."""
    # The input values are from examples in the following textbook:
    # The Complete guide to Option Pricing Formulas, 2nd Edition, Page 154
    price = tff.black_scholes.barrier_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        barriers=barriers,
        rebates=rebates,
        is_barrier_down=is_barrier_down,
        is_knock_out=is_knock_out,
        is_call_options=is_call_options)
    self.assertAllClose(price, expected_price, 10e-3)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_barrier_option_dtype(self, dtype):
    """Function tests barrier option pricing for with given data type."""
    spots = 100.0
    rebates = 3.0
    expiries = 0.5
    discount_rates = 0.08
    dividend_rates = 0.04
    strikes = 90.0
    barriers = 95.0
    expected_price = 9.0246
    is_call_options = True
    is_barrier_down = True
    is_knock_out = True
    volatilities = 0.25
    price = tff.black_scholes.barrier_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        barriers=barriers,
        rebates=rebates,
        is_barrier_down=is_barrier_down,
        is_knock_out=is_knock_out,
        is_call_options=is_call_options,
        dtype=dtype)
    self.assertAllClose(price, expected_price, 10e-3)
    self.assertEqual(price.dtype, dtype)

  def barrier_option_call_xla(self):
    """Tests barrier option price with XLA."""
    dtype = tf.float64
    spots = tf.convert_to_tensor(100.0, dtype=dtype)
    rebates = tf.convert_to_tensor(3.0, dtype=dtype)
    expiries = tf.convert_to_tensor(0.5, dtype=dtype)
    discount_rates = tf.convert_to_tensor(0.08, dtype=dtype)
    dividend_rates = tf.convert_to_tensor(0.04, dtype=dtype)
    strikes = tf.convert_to_tensor(90.0, dtype=dtype)
    barriers = tf.convert_to_tensor(95.0, dtype=dtype)
    expected_price = tf.convert_to_tensor(9.0246, dtype=dtype)
    is_call_options = tf.convert_to_tensor(True)
    is_barrier_down = tf.convert_to_tensor(True)
    is_knock_out = tf.convert_to_tensor(True)
    volatilities = tf.convert_to_tensor(0.25, dtype=dtype)

    def price_barriers_option(samples):
      return tff.black_scholes.barrier_price(
          volatilities=samples[0],
          strikes=samples[1],
          expiries=samples[2],
          spots=samples[3],
          discount_rates=samples[3],
          dividend_rates=samples[4],
          barriers=samples[5],
          rebates=samples[6],
          is_barrier_down=samples[7],
          is_knock_out=samples[8],
          is_call_options=samples[9])[0]

    def xla_compiled_op(samples):
      return tf.function(price_barriers_option, jit_compile=True)(samples)

    price = xla_compiled_op([
        volatilities, strikes, expiries, spots, discount_rates,
        dividend_rates, barriers, rebates, is_barrier_down, is_knock_out,
        is_call_options
    ])
    self.assertAllClose(price, expected_price, 10e-3)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NormalModel',
          'is_normal_model': True,
          'volatilities': [0.01, 0.005],
          'expected_price': [0.3458467885511461, 0.3014786656395892],
      }, {
          'testcase_name': 'LognormalModel',
          'is_normal_model': False,
          'volatilities': [1.0, 0.5],
          'expected_price': [0.34885593, 0.31643427],
      })
  def test_swaption_price(self, is_normal_model, volatilities, expected_price):
    """Function tests swaption pricing."""
    dtype = tf.float64

    expiries = [1.0, 1.0]
    float_leg_start_times = [[1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0],
                             [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]]
    float_leg_end_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                           [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
    fixed_leg_payment_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                               [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
    float_leg_daycount_fractions = [[
        0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0
    ], [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
    fixed_leg_daycount_fractions = [[
        0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0
    ], [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
    fixed_leg_coupon = [0.011, 0.011]
    discount_fn = lambda x: np.exp(-0.01 * np.array(x))
    price = self.evaluate(
        tff.black_scholes.swaption_price(
            volatilities=volatilities,
            expiries=expiries,
            floating_leg_start_times=float_leg_start_times,
            floating_leg_end_times=float_leg_end_times,
            fixed_leg_payment_times=fixed_leg_payment_times,
            floating_leg_daycount_fractions=float_leg_daycount_fractions,
            fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
            fixed_leg_coupon=fixed_leg_coupon,
            floating_leg_start_times_discount_factors=discount_fn(
                float_leg_start_times),
            floating_leg_end_times_discount_factors=discount_fn(
                float_leg_end_times),
            fixed_leg_payment_times_discount_factors=discount_fn(
                fixed_leg_payment_times),
            is_normal_volatility=is_normal_model,
            notional=100.,
            dtype=dtype))

    self.assertAllClose(price, expected_price, 1e-6)


if __name__ == '__main__':
  tf.test.main()
