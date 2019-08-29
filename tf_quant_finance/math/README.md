# Ops for numerical methods.

This module provides ops to perform various mathematical tasks commonly needed
when building quantitative finance models. We do not aim to provide
exhaustive coverage here. [Tensorflow](https://github.com/tensorflow/tensorflow)
and [Tensorflow Probability](https://github.com/tensorflow/probability) provide
a significant suite of methods already and the methods here are meant to
build on those.

The modules provided are:

  * [math.interpolation](interpolation): Ops to perform linear and
  cubic interpolation.
  * [math.pde](pde): Ops to numerically solve partial differential
  equations using finite difference methods. Currently, only linear second
  order PDEs are supported as this is the most commonly needed case.
  * [math.random](random): Ops to compute low discrepancy sequences.


