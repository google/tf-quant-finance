<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.rates.swap_curve_fit" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.rates.swap_curve_fit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/rates/swap_curve_fit.py">View source</a>



Constructs the zero swap curve using optimization.

```python
tf_quant_finance.rates.swap_curve_fit(
    float_leg_start_times, float_leg_end_times, float_leg_daycount_fractions,
    fixed_leg_start_times, fixed_leg_end_times, fixed_leg_daycount_fractions,
    fixed_leg_cashflows, present_values, present_values_settlement_times=None,
    float_leg_discount_rates=None, float_leg_discount_times=None,
    fixed_leg_discount_rates=None, fixed_leg_discount_times=None, optimize=None,
    curve_interpolator=None, initial_curve_rates=None, instrument_weights=None,
    curve_tolerance=1e-08, maximum_iterations=50, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

A zero swap curve is a function of time which gives the interest rate that
can be used to project forward rates at arbitrary `t` for the valuation of
interest rate securities.

Suppose we have a set of `N` Interest Rate Swaps (IRS) `S_i` with increasing
expiries whose market prices are known.
Suppose also that the `i`th IRS issues cashflows at times `T_{ij}` where
`1 <= j <= n_i` and `n_i` is the number of cashflows (including expiry)
for the `i`th swap.
Denote by `T_i` the time of final payment for the `i`th swap
(hence `T_i = T_{i,n_i}`). This function estimates a set of rates `r(T_i)`
such that when these rates are interpolated to all other cashflow times,
the computed value of the swaps matches the market value of the swaps
(within some tolerance). Rates at intermediate times are interpolated using
the user specified interpolation method (the default interpolation method
is linear interpolation on rates).

#### Example:

The following example illustrates the usage by building an implied swap curve
from four vanilla (fixed to float) LIBOR swaps.

```python

dtype = np.float64

# Next we will set up LIBOR reset and payment times for four spot starting
# swaps with maturities 1Y, 2Y, 3Y, 4Y. The LIBOR rate spans 6M.

float_leg_start_times = [
          np.array([0., 0.5], dtype=dtype),
          np.array([0., 0.5, 1., 1.5], dtype=dtype),
          np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5], dtype=dtype),
          np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=dtype)
      ]

float_leg_end_times = [
          np.array([0.5, 1.0], dtype=dtype),
          np.array([0.5, 1., 1.5, 2.0], dtype=dtype),
          np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
          np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
      ]

# Next we will set up start and end times for semi-annual fixed coupons.

fixed_leg_start_times = [
          np.array([0., 0.5], dtype=dtype),
          np.array([0., 0.5, 1., 1.5], dtype=dtype),
          np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5], dtype=dtype),
          np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=dtype)
      ]

fixed_leg_end_times = [
          np.array([0.5, 1.0], dtype=dtype),
          np.array([0.5, 1., 1.5, 2.0], dtype=dtype),
          np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
          np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
      ]

# Next setup a trivial daycount for floating and fixed legs.

float_leg_daycount = [
          np.array([0.5, 0.5], dtype=dtype),
          np.array([0.5, 0.5, 0.5, 0.5], dtype=dtype),
          np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype),
          np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype)
      ]

fixed_leg_daycount = [
          np.array([0.5, 0.5], dtype=dtype),
          np.array([0.5, 0.5, 0.5, 0.5], dtype=dtype),
          np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype),
          np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype)
      ]

fixed_leg_cashflows = [
      # 1 year swap with 2.855% semi-annual fixed payments.
      np.array([-0.02855, -0.02855], dtype=dtype),
      # 2 year swap with 3.097% semi-annual fixed payments.
      np.array([-0.03097, -0.03097, -0.03097, -0.03097], dtype=dtype),
      # 3 year swap with 3.1% semi-annual fixed payments.
      np.array([-0.031, -0.031, -0.031, -0.031, -0.031, -0.031], dtype=dtype),
      # 4 year swap with 3.2% semi-annual fixed payments.
      np.array([-0.032, -0.032, -0.032, -0.032, -0.032, -0.032, -0.032,
      -0.032], dtype=dtype)
  ]

# The present values of the above IRS.
pvs = np.array([0., 0., 0., 0.], dtype=dtype)

# Initial state of the curve.
initial_curve_rates = np.array([0.01, 0.01, 0.01, 0.01], dtype=dtype)

results = swap_curve_fit(float_leg_start_times, float_leg_end_times,
                         float_leg_daycount, fixed_leg_start_times,
                         fixed_leg_end_times, fixed_leg_cashflows,
                         fixed_leg_daycount, pvs, dtype=dtype,
                         initial_curve_rates=initial_curve_rates)

#### References:
[1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
    Volume I: Foundations and Vanilla Models. Chapter 6. 2010.

#### Args:


* <b>`float_leg_start_times`</b>: List of `Tensor`s. Each `Tensor` must be of rank 1
  and of the same real dtype. They may be of different sizes. Each `Tensor`
  represents the beginning of the accrual period for the forward rate which
  determines the floating payment. Each element in the list belong to a
  unique swap to be used to build the curve.
* <b>`float_leg_end_times`</b>: List of `Tensor`s. Each `Tensor` must be of rank 1 and
  and the same shape and of the same real dtype as the corresponding element
  in `float_leg_start_times`. Each `Tensor` represents the end of the
  accrual period for the forward rate which determines the floating payment.
* <b>`float_leg_daycount_fractions`</b>: List of `Tensor`s. Each `Tensor` must be of
  the same shape and type as `float_leg_start_times`. They may be of
  different sizes. Each `Tensor` represents the daycount fraction of the
  forward rate which determines the floating payment.
* <b>`fixed_leg_start_times`</b>: List of `Tensor`s. Each `Tensor` must be of rank 1
  and of the same real dtype. They may be of different sizes. Each `Tensor`
  represents the begining of the accrual period fixed coupon.
* <b>`fixed_leg_end_times`</b>: List of `Tensor`s. Each `Tensor` must be of the same
  shape and type as `fixed_leg_start_times`. Each `Tensor` represents the
  end of the accrual period for the fixed coupon.
* <b>`fixed_leg_daycount_fractions`</b>: List of `Tensor`s. Each `Tensor` must be of
  the same shape and type as `fixed_leg_start_times`. Each `Tensor`
  represents the daycount fraction applicable for the fixed payment.
* <b>`fixed_leg_cashflows`</b>: List of `Tensor`s. The list must be of the same length
  as the `fixed_leg_start_times`. Each `Tensor` must be of rank 1 and of the
  same dtype as the `Tensor`s in `fixed_leg_start_times`. The input contains
  fixed cashflows at each coupon payment time including notional (if any).
  The sign should be negative (positive) to indicate net outgoing (incoming)
  cashflow.
* <b>`present_values`</b>: List containing scalar `Tensor`s of the same dtype as
  elements of `fixed_leg_cashflows`. The length of the list must be the same
  as the length of `fixed_leg_cashflows`. The input contains the market
  price of the underlying instruments.
* <b>`present_values_settlement_times`</b>: List containing scalar `Tensor`s of the
  same dtype as elements of `present_values`. The length of the list must be
  the same as the length of `present_values`. The settlement times for the
  present values is the time from now when the instrument is traded to the
  time that the purchase price is actually delivered. If not supplied, then
  it is assumed that the settlement times are zero for every bond.
  Default value: `None` which is equivalent to zero settlement times.
* <b>`float_leg_discount_rates`</b>: Optional `Tensor` of the same dtype as
  `initial_discount_rates`. This input contains the continuously compounded
  discount rates the will be used to discount the floating cashflows. This
  allows the swap curve to constructed using an independent discount curve
  (e.g. OIS curve). By default the cashflows are discounted using the curve
  that is being constructed.
* <b>`float_leg_discount_times`</b>: Optional `Tensor` of the same dtype and shape as
  `float_leg_discount_rates`. This input contains the times corresponding to
  the rates specified via the `float_leg_discount_rates`.
* <b>`fixed_leg_discount_rates`</b>: Optional `Tensor` of the same dtype as
  `initial_discount_rates`. This input contains the continuously compounded
  discount rates the will be used to discount the fixed cashflows. This
  allows the swap curve to constructed using an independent discount curve
  (e.g. OIS curve). By default the cashflows are discounted using the curve
  that is being constructed.
* <b>`fixed_leg_discount_times`</b>: Optional `Tensor` of the same dtype and shape as
  `fixed_leg_discount_rates`. This input contains the times corresponding to
  the rates specified via the `fixed_leg_discount_rates`.
* <b>`optimize`</b>: Optional Python callable which implements the algorithm used to
  minimize the objective function during curve construction. It should have
  the following interface:
  result = optimize(value_and_gradients_function, initial_position,
  tolerance, max_iterations)
  `value_and_gradients_function` is a Python callable that accepts a point
  as a real `Tensor` and returns a tuple of `Tensor`s of real dtype
  containing the value of the function and its gradient at that point.
  'initial_position' is a real `Tensor` containing the starting point of the
  optimization, 'tolerance' is a real scalar `Tensor` for stopping tolerance
  for the procedure and `max_iterations` specifies the maximum number of
  iterations.
  `optimize` should return a namedtuple containing the items: `position` (a
  tensor containing the optimal value), `converged` (a boolean indicating
  whether the optimize converged according the specified criteria),
  `failed` (a boolean indicating if the optimization resulted in a failure),
  `num_iterations` (the number of iterations used), and `objective_value` (
  the value of the objective function at the optimal value).
  The default value for `optimize` is None and conjugate gradient algorithm
  is used.
* <b>`curve_interpolator`</b>: Optional Python callable used to interpolate the zero
  swap rates at cashflow times. It should have the following interface:
  yi = curve_interpolator(xi, x, y)
  `x`, `y`, 'xi', 'yi' are all `Tensors` of real dtype. `x` and `y` are the
  sample points and values (respectively) of the function to be
  interpolated. `xi` are the points at which the interpolation is
  desired and `yi` are the corresponding interpolated values returned by the
  function. The default value for `curve_interpolator` is None in which
  case linear interpolation is used.
* <b>`initial_curve_rates`</b>: Optional `Tensor` of the same dtype and shape as
  `present_values`. The starting guess for the discount rates used to
  initialize the iterative procedure.
  Default value: `None`. If not supplied, the yields to maturity for the
    bonds is used as the initial value.
* <b>`instrument_weights`</b>: Optional 'Tensor' of the same dtype and shape as
  `present_values`. This input contains the weight of each instrument in
  computing the objective function for the conjugate gradient optimization.
  By default the weights are set to be the inverse of maturities.
* <b>`curve_tolerance`</b>: Optional positive scalar `Tensor` of same dtype as
  elements of `bond_cashflows`. The absolute tolerance for terminating the
  iterations used to fit the rate curve. The iterations are stopped when the
  estimated discounts at the expiry times of the bond_cashflows change by a
  amount smaller than `discount_tolerance` in an iteration.
  Default value: 1e-8.
* <b>`maximum_iterations`</b>: Optional positive integer `Tensor`. The maximum number
  of iterations permitted when fitting the curve.
  Default value: 50.
* <b>`dtype`</b>: `tf.Dtype`. If supplied the dtype for the (elements of)
  `float_leg_start_times`, and `fixed_leg_start_times`.
  Default value: None which maps to the default dtype inferred by
  TensorFlow.
* <b>`name`</b>: Python str. The name to give to the ops created by this function.
  Default value: `None` which maps to 'swap_curve'.


#### Returns:


* <b>`curve_builder_result`</b>: An instance of `SwapCurveBuilderResult` containing the
  following attributes.
  times: Rank 1 real `Tensor`. Times for the computed discount rates. These
    are chosen to be the expiry times of the supplied cashflows.
  discount_rates: Rank 1 `Tensor` of the same dtype as `times`.
    The inferred discount rates.
  discount_factor: Rank 1 `Tensor` of the same dtype as `times`.
    The inferred discount factors.
  initial_discount_rates: Rank 1 `Tensor` of the same dtype as `times`. The
    initial guess for the discount rates.
  converged: Scalar boolean `Tensor`. Whether the procedure converged.
    The procedure is said to have converged when the maximum absolute
    difference in the discount factors from one iteration to the next falls
    below the `discount_tolerance`.
  failed: Scalar boolean `Tensor`. Whether the procedure failed. Procedure
    may fail either because a NaN value was encountered for the discount
    rates or the discount factors.
  iterations: Scalar int32 `Tensor`. Number of iterations performed.
  objective_value: Scalar real `Tensor`. The value of the ibjective function
    evaluated using the fitted swap curve.


#### Raises:


* <b>`ValueError`</b>: If the initial state of the curve is not
supplied to the function.