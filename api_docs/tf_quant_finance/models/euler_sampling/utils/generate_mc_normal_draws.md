<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.models.euler_sampling.utils.generate_mc_normal_draws" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.models.euler_sampling.utils.generate_mc_normal_draws

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/utils.py">View source</a>



Generates normal random samples to be consumed by a Monte Carlo algorithm.

```python
tf_quant_finance.models.euler_sampling.utils.generate_mc_normal_draws(
    num_normal_draws, num_time_steps, num_sample_paths, random_type, skip=0,
    seed=None, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

Many of Monte Carlo (MC) algorithms can be re-written so that all necessary
random (or quasi-random) variables are drawn in advance as a `Tensor` of
shape `[num_time_steps, num_samples, num_normal_draws]`, where
`num_time_steps` is the number of time steps Monte Carlo algorithm performs,
`num_sample_paths` is a number of sample paths of the Monte Carlo algorithm
and `num_normal_draws` is a number of independent normal draws per sample
paths.
For example, in order to use quasi-random numbers in a Monte Carlo algorithm,
the samples have to be drawn in advance.
The function generates a `Tensor`, say, `x` in a format such that for a
quasi-`random_type` `x[i]` is correspond to different dimensions of the
quasi-random sequence, so that it can be used in a Monte Carlo algorithm

#### Args:


* <b>`num_normal_draws`</b>: A scalar int32 `Tensor`. The number of independent normal
  draws at each time step for each sample path. Should be a graph
  compilation constant.
* <b>`num_time_steps`</b>: A scalar int32 `Tensor`. The number of time steps at which
  to draw the independent normal samples. Should be a graph compilation
  constant.
* <b>`num_sample_paths`</b>: A scalar int32 `Tensor`. The number of trajectories (e.g.,
  Monte Carlo paths) for which to draw the independent normal samples.
  Should be a graph compilation constant.
* <b>`random_type`</b>: Enum value of `tff.math.random.RandomType`. The type of
  (quasi)-random number generator to use to generate the paths.
* <b>`skip`</b>: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
  Halton sequence to skip. Used only when `random_type` is 'SOBOL',
  'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
  Default value: `0`.
* <b>`seed`</b>: Seed for the random number generator. The seed is only relevant if
  `random_type` is one of `[PSEUDO, PSEUDO_ANTITHETIC, HALTON_RANDOMIZED]`.
* <b>`dtype`</b>: The `dtype` of the output `Tensor`.
  Default value: `None` which maps to `float32`.
* <b>`name`</b>: Python string. The name to give this op.
  Default value: `None` which maps to `generate_mc_normal_draws`.


#### Returns:

A `Tensor` of shape `[num_time_steps, num_sample_paths, num_normal_draws]`.
