# Random and Low Discrepancy Sequences.

This directory contains ops to efficiently generate random numbers and
quasi-random low discrepancy sequences.

For the random numbers, [TensorFlow](tensorflow.org) and
[TensorFlow Probability](https://www.tensorflow.org/probability)
already contain significant support. This module provides Sobol and Halton
low discrepancy sequences as well as a multivariate normal sampler which
supports using these sequences (i.e. draws from a gaussian copula with low
discrepancy sequences).
