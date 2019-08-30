# High-performance TensorFlow library for quantitative finance

This library provides high-performance methods leveraging the hardware
acceleration support and automatic differentation of TensorFlow. The initial
library will provide TensorFlow support for foundational mathematical methods,
mid-level methods, and specific pricing models. These will be significantly
expanded over next months.

Our TensorFlow-based methods will be structured along three tiers:

1. **Foundational methods**.
Core mathematics methods - optimisation, root finders, linear algebra, random number generation, etc.

2. **Mid-level methods**.
ODE & PDE solvers, Diffusion Path Generators, Copula samplers etc.

3. **Pricing methods**.
Specific Pricing models (e.g Local Vol (LV), Stochastic Vol (SV), Stochastic Local Vol (SLV), Hull-White (HW))

## Installation

The easiest way to get started with the library is via the pip package.

```sh
pip install tf_quant_finance
```

## Development Roadmap

We are working on expanding the coverage of the library. Areas under active
development are:

  * Ito Processes: Framework for defining [Ito processes](https://en.wikipedia.org/wiki/It%C3%B4_calculus#It%C3%B4_processes).
  Includes methods for sampling paths from a process and for solving the
  associated backward Kolmogorov equation.
  * Implementation of the following specific processes/models:
      * Brownian Motion
      * Geometric Brownian Motion
      * Ornstein-Uhlenbeck
      * Single factor Hull White model
      * Heston model
      * Local volatility model.
      * SABR model
  * ADI method for solving multi dimensional PDEs.
  * Copulas: Support for defining and sampling from copulas.
  * Model Calibration:
      * Dupire local vol calibration.
      * SABR model calibration.
  * Rate curve fitting: Hagan-West algorithm for yield curve bootstrapping and
  the Monotone Convex interpolation scheme.
  * Optimization:
      * Conjugate gradient optimizer.


## Examples
Our existing examples will be made available here soon.

## Contributing
We're eager to collaborate with you! See [CONTRIBUTING.md](CONTRIBUTING.md) for a guide on how to contribute. This project adheres to TensorFlow's code of conduct. By participating, you are expected to uphold this code.

## For developers

This section is meant for developers who want to contribute code to the
library. If you are only interested in using the library, please follow the
instructions in the [Installation](#installation) section.

### Dependencies

This library has the following dependencies:

1.  Bazel
2.  Python 3 (Bazel uses Python 3 by default)
3.  TensorFlow
4.  TensorFlow Probability
5.  Numpy

This library requires the
[Bazel](https://bazel.build/) build system. Please follow the
[Bazel installation instructions](https://docs.bazel.build/versions/master/install.html)
for your platform.


You can install TensorFlow and related dependencies using the ```pip3 install```
command:

```sh
pip3 install --upgrade tensorflow tensorflow_probability numpy
```

### Commonly used commands

Clone the GitHub repository:

```sh
git clone https://github.com/google/tf-quant-finance.git
```

After you run

```sh
cd tf_quant_finance
```

you can execute tests using the ```bazel test``` command. For example,

```sh
bazel test tf_quant_finance/math/random/sobol:sobol_test
```

will run tests in
[sobol_test.py](https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/math/random/sobol/sobol_test.py)
.

Tests will be run using the Python version 3. Please make sure that you can
run ```import tensorflow``` in the Python 3 shell, otherwise tests might fail.

### Building a custom pip package

The following commands will build custom pip package from source and install it:

```sh
# sudo apt-get install bazel git python python-pip rsync # For Ubuntu.
git clone https://github.com/google/tf-quant-finance.git
cd tf-quant-finance
bazel build :build_pip_pkg
./bazel-bin/build_pip_pkg artifacts
pip install --user --upgrade artifacts/*.whl
```

## Community
1. [GitHub repository](https://github.com/google/tf-quant-finance): Report bugs or make feature requests.

2. [TensorFlow Blog](https://medium.com/tensorflow): Stay up to date on content from the TensorFlow team and best articles from the community.

3. tf-quant-finance@google.com: Open mailing list for discussion and questions of this library.

4. TensorFlow Probability: This library will leverage methods from [TensorFlow Probability](https://www.tensorflow.org/probability) (TFP).

## Disclaimers
This is not an officially supported Google product. This library is under active development. Interfaces may change at any time.

## License
This library is licensed under the Apache 2 license (see [LICENSE](LICENSE)). This library uses Sobol primitive polynomials and initial direction numbers
which are licensed under the BSD license.
