<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.models.euler_sampling.utils.maybe_update_along_axis" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.models.euler_sampling.utils.maybe_update_along_axis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/utils.py">View source</a>



Replace `tensor` entries with `new_tensor` along a given axis.

```python
tf_quant_finance.models.euler_sampling.utils.maybe_update_along_axis(
    *, tensor, new_tensor, axis, ind, do_update, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

This updates elements of `tensor` that correspond to the elements returned by
`numpy.take(updated, ind, axis)` with the corresponding elements of
`new_tensor`.

# Example
```python
tensor = tf.ones([5, 4, 3, 2])
new_tensor = tf.zeros([5, 4, 3, 2])
updated_tensor = maybe_update_along_axis(tensor=tensor,
                                         new_tensor=new_tensor,
                                         axis=1,
                                         ind=2,
                                         do_update=True)
# Returns a `Tensor` of ones where
# `updated_tensor[:, 2, :, :].numpy() == 0`
```
If the `do_update` is set to `False`, then the update does not happen unless
the number of dimensions along the `axis` is equal to 1. This functionality
is useful when, for example, aggregating samples of an Ito process.

#### Args:


* <b>`tensor`</b>: A `Tensor` of any shape and `dtype`.
* <b>`new_tensor`</b>: A `Tensor` of the same `dtype` as `tensor` and of shape
  broadcastable with `tensor`.
* <b>`axis`</b>: A Python integer. The axis of `tensor` along which the elements have
  to be updated.
* <b>`ind`</b>: An int32 scalar `Tensor` that denotes an index on the `axis` which
  defines the updated slice of `tensor` (see example above).
* <b>`do_update`</b>: A bool scalar `Tensor`. If `False`, the output is the same as
  `tensor`, unless  the dimension of the `tensor` along the `axis` is equal
  to 1.
* <b>`dtype`</b>: The `dtype` of the input `Tensor`s.
  Default value: `None` which means that default dtypes inferred by
    TensorFlow are used.
* <b>`name`</b>: Python string. The name to give this op.
  Default value: `None` which maps to `maybe_update_along_axis`.


#### Returns:

A `Tensor` of the same shape and `dtype` as `tensor`.
