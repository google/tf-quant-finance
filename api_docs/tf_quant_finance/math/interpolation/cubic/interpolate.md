<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.interpolation.cubic.interpolate" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.interpolation.cubic.interpolate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/math/interpolation/cubic/cubic_interpolation.py">View source</a>



Interpolates spline values for the given `x_values` and the `spline_data`.

```python
tf_quant_finance.math.interpolation.cubic.interpolate(
    x_values, spline_data, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

Constant extrapolation is performed for the values outside the domain
`spline_data.x_data`. This means that for `x > max(spline_data.x_data)`,
`interpolate(x, spline_data) = spline_data.y_data[-1]`
and for  `x < min(spline_data.x_data)`,
`interpolate(x, spline_data) = spline_data.y_data[0]`.

For the interpolation formula refer to p.548 of [1].

#### References:
[1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.
  Link: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf

#### Args:


* <b>`x_values`</b>: A real `Tensor` of shape `batch_shape + [num_points]`.
* <b>`spline_data`</b>: An instance of `SplineParameters`. `spline_data.x_data` should
  have the same batch shape as `x_values`.
* <b>`dtype`</b>: Optional dtype for `x_values`.
  Default value: `None` which maps to the default dtype inferred by
  TensorFlow.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` which is mapped to the default name
  `cubic_spline_interpolate`.


#### Returns:

A `Tensor` of the same shape and `dtype` as `x_values`. Represents
the interpolated values.



#### Raises:


* <b>`ValueError`</b>:   If `x_values` batch shape is different from `spline_data.x_data` batch
  shape.