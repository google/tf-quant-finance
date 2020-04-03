<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.models.euler_sampling.sample" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.models.euler_sampling.sample

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/euler_sampling.py">View source</a>



Returns a sample paths from the process using Euler method.

```python
tf_quant_finance.models.euler_sampling.sample(
    dim, drift_fn, volatility_fn, times, time_step, num_samples=1,
    initial_state=None, random_type=None, seed=None, swap_memory=True, skip=0,
    dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

For an Ito process,

```
  dX = a(t, X_t) dt + b(t, X_t) dW_t
```
with given drift `a` and volatility `b` functions Euler method generates a
sequence {X_n} as

```
X_{n+1} = X_n + a(t_n, X_n) dt + b(t_n, X_n) (N(0, t_{n+1}) - N(0, t_n)),
```
where `dt = t_{n+1} - t_n` and `N` is a sample from the Normal distribution.
See [1] for details.

#### References
[1]: Wikipedia. Euler-Maruyama method:
https://en.wikipedia.org/wiki/Euler-Maruyama_method

#### Args:


* <b>`dim`</b>: Python int greater than or equal to 1. The dimension of the Ito
  Process.
* <b>`drift_fn`</b>: A Python callable to compute the drift of the process. The
  callable should accept two real `Tensor` arguments of the same dtype.
  The first argument is the scalar time t, the second argument is the
  value of Ito process X - tensor of shape `batch_shape + [dim]`.
  The result is value of drift a(t, X). The return value of the callable
  is a real `Tensor` of the same dtype as the input arguments and of shape
  `batch_shape + [dim]`.
* <b>`volatility_fn`</b>: A Python callable to compute the volatility of the process.
  The callable should accept two real `Tensor` arguments of the same dtype
  and shape `times_shape`. The first argument is the scalar time t, the
  second argument is the value of Ito process X - tensor of shape
  `batch_shape + [dim]`. The result is value of drift b(t, X). The return
  value of the callable is a real `Tensor` of the same dtype as the input
  arguments and of shape `batch_shape + [dim, dim]`.
* <b>`times`</b>: Rank 1 `Tensor` of increasing positive real values. The times at
  which the path points are to be evaluated.
* <b>`time_step`</b>: Scalar real `Tensor` - maximal distance between points
    in grid in Euler schema.
* <b>`num_samples`</b>: Positive scalar `int`. The number of paths to draw.
  Default value: 1.
* <b>`initial_state`</b>: `Tensor` of shape `[dim]`. The initial state of the
  process.
  Default value: None which maps to a zero initial state.
* <b>`random_type`</b>: Enum value of `RandomType`. The type of (quasi)-random
  number generator to use to generate the paths.
  Default value: None which maps to the standard pseudo-random numbers.
* <b>`seed`</b>: Python `int`. The random seed to use.
  Default value: None, which  means no seed is set.
* <b>`swap_memory`</b>: A Python bool. Whether GPU-CPU memory swap is enabled for this
  op. See an equivalent flag in `tf.while_loop` documentation for more
  details. Useful when computing a gradient of the op since `tf.while_loop`
  is used to propagate stochastic process in time.
  Default value: True.
* <b>`skip`</b>: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
  Halton sequence to skip. Used only when `random_type` is 'SOBOL',
  'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
  Default value: `0`.
* <b>`dtype`</b>: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
  Default value: None which means that the dtype implied by `times` is
  used.
* <b>`name`</b>: Python string. The name to give this op.
  Default value: `None` which maps to `euler_sample`.


#### Returns:

A real `Tensor` of shape [num_samples, k, n] where `k` is the size of the
   `times`, `n` is the dimension of the process.



#### Raises:


* <b>`ValueError`</b>: If `time_step` or `times` have a non-constant value (e.g.,
  values are random), and `random_type` is `SOBOL`. This will be fixed with
  the release of TensorFlow 2.2.