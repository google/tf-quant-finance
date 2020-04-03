<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.interpolation.cubic.build_spline" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.interpolation.cubic.build_spline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/math/interpolation/cubic/cubic_interpolation.py">View source</a>



Builds a SplineParameters interpolation object.

```python
tf_quant_finance.math.interpolation.cubic.build_spline(
    x_data, y_data, validate_args=False, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

Given a `Tensor` of state points `x_data` and corresponding values `y_data`
creates an object that contains iterpolation coefficients. The object can be
used by the `interpolate` function to get interpolated values for a set of
state points `x` using the cubic spline interpolation algorithm.
It assumes that the second derivative at the first and last spline points
are zero. The basic logic is explained in [1] (see also, e.g., [2]).

Repeated entries in `x_data` are allowed for the boundary values of `x_data`.
For example, `x_data` can be `[1., 1., 2, 3. 4., 4., 4.]` but not
`[1., 2., 2., 3.]`. The repeated values play no role in interpolation and are
useful only for interpolating multiple splines with different numbers of data
point. It is user responsibility to verify that the corresponding
values of `y_data` are the same for the repeated values of `x_data`.

#### Typical Usage Example:



```python
import tensorflow.compat.v2 as tf
import numpy as np

x_data = np.linspace(-5.0, 5.0,  num=11)
y_data = 1.0/(1.0 + x_data**2)
spline = cubic_interpolation.build(x_data, y_data)
x_args = [3.3, 3.4, 3.9]

y = cubic_interpolation.interpolate(x_args, spline)
```

#### References:
[1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.
  Link: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf
[2]: R. Pienaar, M Choudhry. Fitting the term structure of interest rates:
  the practical implementation of cubic spline methodology.
  Link:
  http://yieldcurve.com/mktresearch/files/PienaarChoudhry_CubicSpline2.pdf

#### Args:


* <b>`x_data`</b>: A real `Tensor` of shape `[..., num_points]` containing
  X-coordinates of points to fit the splines to. The values have to
  be monotonically non-decreasing along the last dimension.
* <b>`y_data`</b>: A `Tensor` of the same shape and `dtype` as `x_data` containing
  Y-coordinates of points to fit the splines to.
* <b>`validate_args`</b>: Python `bool`. When `True`, verifies if elements of `x_data`
  are sorted in the last dimension in non-decreasing order despite possibly
  degrading runtime performance.
  Default value: False.
* <b>`dtype`</b>: Optional dtype for both `x_data` and `y_data`.
  Default value: `None` which maps to the default dtype inferred by
  TensorFlow.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` which is mapped to the default name
  `cubic_spline_build`.


#### Returns:

An instance of `SplineParameters`.
