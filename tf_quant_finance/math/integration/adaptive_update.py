"""Update function for intervals in adaptive numeric integration."""
from typing import Optional
import tensorflow.compat.v2 as tf

from tf_quant_finance import types


def update(
    lower: types.FloatTensor,
    upper: types.FloatTensor,
    estimate: types.FloatTensor,
    error: types.FloatTensor,
    tolerance: float,
    dtype: Optional[tf.DType] = None,
    name: Optional[str] = None
) -> (types.FloatTensor, types.FloatTensor, types.FloatTensor):
  """Calculates new values for the limits for any adaptive quadrature.

  Checks which intervals have estimated results that are within the provided
  tolerance. The values for these intervals are added to the sum of good
  estimations. The other intervals get divided in half.

  #### Example
  ```python
    l = tf.constant([[[0.0], [1.0]]])
    u = tf.constant([[[1.0], [2.0]]])
    estimate = tf.constant([[[3.0], [4.0]]])
    err = tf.constant([[[0.01], [0.02]]])
    tol = 0.004
    update(l, u, estimate, err, tol)
    # tf.constant([[1.0, 1.5]]), tf.constant([[1.5, 2.0]]), tf.constant([3.0])
  ```

  Args:
    lower: Represents the lower limits of integration. Must be a 2-dimensional
      tensor of shape `[batch_dim, n]` (where `n` is defined by the algorithm
      and represents the number of subintervals).
    upper: Same shape and dtype as `lower` representing the upper limits of
      intergation.
    estimate: Same shape and dtype as `lower` representing the integration
      results calculated with some quadrature method for the corresponding
      limits.
    error: Same shape and dtype as `lower` representing the estimated
      integration error for corresponding `estimate` values.
    tolerance: Represents the tolerance for the estimated error of the integral
      estimation, at which to stop further dividing the intervals.
    dtype: If supplied, the dtype for the `lower` and `upper`. Result will have
      the same dtype. Default value: None which maps to dtype of `lower`.
    name: The name to give to the ops created by this function. Default value:
      None which maps to 'adaptive_update'.

  Returns:
    A tuple:
      * `Tensor` of shape `[batch_dim, new_n]`, containing values of the new
      lower limits,
      * `Tensor` of shape `[batch_dim, new_n]`, containing values of the new
      upper limits,
      * `Tensor` of shape `[batch_dim]`, containing sum values of the quadrature
      method results of the good intervals.
  """
  with tf.name_scope(name=name or 'adaptive_update'):
    # Shape [batch_dim, n]
    lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
    dtype = lower.dtype
    upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
    relative_error = error / estimate
    condition = (relative_error > tolerance)
    # To have matching dimensions we keep the same number of sub-intervals from
    # each batch element.
    # We do this by finding the maximum number of sub-intervals needed to be
    # re-calculated among all batch elements (k) and getting the top k worst
    # sub-intervals from all batch elements.
    # count max number of True values along batch_dim
    num_bad_sub_intervals = tf.reduce_max(
        tf.math.count_nonzero(condition, axis=1, dtype=tf.int32), axis=0)
    # get indices where the error ration is in the top 'num_bad_sub_intervals'
    # Shape [batch_dim, num_bad_sub_intervals]
    indices = tf.math.top_k(
        relative_error, k=num_bad_sub_intervals, sorted=False).indices

    # calculate sum of good estimates
    # Shape [batch_dim]
    sum_all = tf.reduce_sum(estimate, axis=-1)
    sum_bad = tf.reduce_sum(
        tf.gather(estimate, indices, batch_dims=-1), axis=-1)
    sum_goods = sum_all - sum_bad

    # calculate new upper and lower bounds
    # Shape [batch_dim, num_bad_sub_intervals]
    filtered_lower = tf.gather(lower, indices, batch_dims=-1)
    filtered_upper = tf.gather(upper, indices, batch_dims=-1)
    mid_points = (filtered_lower + filtered_upper) / 2
    # Shape [batch_dim, num_bad_sub_intervals * 2]
    new_lower = tf.concat([filtered_lower, mid_points], axis=-1)
    new_upper = tf.concat([mid_points, filtered_upper], axis=-1)

    # new_n = num_bad_sub_intervals * 2
    # Shapes [batch_dim, new_n], [batch_dim, new_n], [batch_dim]
    return new_lower, new_upper, sum_goods
