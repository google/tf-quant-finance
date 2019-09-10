# Ops for numerical methods.

This module provides ops to perform various mathematical tasks commonly needed
when building quantitative finance models. We do not aim to provide
exhaustive coverage here. [Tensorflow](https://github.com/tensorflow/tensorflow)
and [Tensorflow Probability](https://github.com/tensorflow/probability) provide
a significant suite of methods already and the methods here are meant to
build on those.

Some of the modules/functions provided are:

  * [math.interpolation](interpolation): Ops to perform linear and
  cubic interpolation.
  * [math.optimizer](optimizer): Ops for numerical optimization.
  * [math.pde](pde): Ops to numerically solve partial differential
  equations using finite difference methods. Currently, only linear second
  order PDEs are supported as this is the most commonly needed case.
  * [math.random](random): Ops to compute low discrepancy sequences.
  * [math.root_search](root_search.py): Provides the Brent method for computing
    roots of functions in one dimension.
  * [math.segment_ops](segment_ops.py): Utility methods to apply some element
    wise ops in a segment.

