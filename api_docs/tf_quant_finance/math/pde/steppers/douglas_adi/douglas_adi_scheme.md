<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.pde.steppers.douglas_adi.douglas_adi_scheme" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.pde.steppers.douglas_adi.douglas_adi_scheme

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/math/pde/steppers/douglas_adi.py">View source</a>



Applies Douglas time marching scheme (see [1] and Eq. 3.1 in [2]).

```python
tf_quant_finance.math.pde.steppers.douglas_adi.douglas_adi_scheme(
    theta
)
```



<!-- Placeholder for "Used in" -->

Time marching schemes solve the space-discretized equation
`du/dt = A(t) u(t) + b(t)` where `u` and `b` are vectors and `A` is a matrix;
see more details in multidim_parabolic_equation_stepper.py.

In Douglas scheme (as well as other ADI schemes), the matrix `A` is
represented as sum `A = sum_i A_i + A_mixed`. `A_i` is the contribution of
terms with partial derivatives w.r.t. dimension `i`, and `A_mixed` is the
contribution of all the mixed-derivative terms. The shift term is split evenly
between `A_i`. Similarly, inhomogeneous term is represented as sum `b = sum_i
b_i`, where `b_i` comes from boundary conditions on boundary orthogonal to
dimension `i`.

Given the current values vector u(t1), the step is defined as follows
(using the notation of Eq. 3.1 in [2]):
`Y_0 = (1 + A(t1) dt) U_{n-1} + b(t1) dt`,
`Y_j = Y_{j-1} + theta dt (A_j(t2) Y_j - A_j(t1) U_{n-1} + b_j(t2) - b_j(t1))`
for each spatial dimension `j`, and
`U_n = Y_{n_dims-1}`.

Here the parameter `theta` is a non-negative number, `U_{n-1} = u(t1)`,
`U_n = u(t2)`, and `dt = t2 - t1`.

Note: Douglas scheme is only first-order accurate if mixed terms are
present. More advanced schemes, such as Craig-Sneyd scheme, are needed to
achieve the second-order accuracy.

#### References:
[1] Douglas Jr., Jim (1962), "Alternating direction methods for three space
  variables", Numerische Mathematik, 4 (1): 41-63
[2] Tinne Haentjens, Karek J. in't Hout. ADI finite difference schemes for
  the Heston-Hull-White PDE. https://arxiv.org/abs/1111.4087

#### Args:


* <b>`theta`</b>: Number between 0 and 1 (see the step definition above). `theta = 0`
  corresponds to fully-explicit scheme.


#### Returns:

A callable consumes the following arguments by keyword:
  1. inner_value_grid: Grid of solution values at the current time of
    the same `dtype` as `value_grid` and shape of `value_grid[..., 1:-1]`.
  2. t1: Time before the step.
  3. t2: Time after the step.
  4. equation_params_fn: A callable that takes a scalar `Tensor` argument
    representing time, and constructs the tridiagonal matrix `A`
    (a tuple of three `Tensor`s, main, upper, and lower diagonals)
    and the inhomogeneous term `b`. All of the `Tensor`s are of the same
    `dtype` as `inner_value_grid` and of the shape broadcastable with the
    shape of `inner_value_grid`.
  5. n_dims: A Python integer, the spatial dimension of the PDE.
The callable returns a `Tensor` of the same shape and `dtype` a
`values_grid` and represents an approximate solution `u(t2)`.
