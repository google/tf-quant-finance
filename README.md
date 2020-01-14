# TF Quant Finance: TensorFlow based Quant Finance Library

[![Build Status](https://travis-ci.org/google/tf-quant-finance.svg?branch=master)](https://travis-ci.org/google/tf-quant-finance)

## Table of contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [TensorFlow training](#tensorflow-training)
4. [Development roadmap](#development-roadmap)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [Development](#development)
8. [Community](#community)
9. [Disclaimers](#disclaimers)
10. [License](#license)

## Introduction

This library provides high-performance components leveraging the hardware
acceleration support and automatic differentiation of TensorFlow. The
library will provide TensorFlow support for foundational mathematical methods,
mid-level methods, and specific pricing models. The coverage is being rapidly
expanded over the next few months.

The library is structured along three tiers:

1. **Foundational methods**.
Core mathematical methods - optimisation, interpolation, root finders,
linear algebra, random and quasi-random number generation, etc.

2. **Mid-level methods**.
ODE & PDE solvers, Ito process framework, Diffusion Path Generators,
Copula samplers etc.

3. **Pricing methods and other quant finance specific utilities**.
Specific Pricing models (e.g Local Vol (LV), Stochastic Vol (SV),
Stochastic Local Vol (SLV), Hull-White (HW)) and their calibration.
Rate curve building and payoff descriptions.

We aim for the library components to be easily accessible at each level. Each
layer will be accompanied by many examples which can be run independently of
higher level components.

## Installation

The easiest way to get started with the library is via the pip package.

Note that library requires Python 3 and Tensorflow >= 2.1

First please install the most recent version of TensorFlow by following
the [TensorFlow installation instructions](https://tensorflow.org/install).
For example, you could install TensorFlow using

```sh
pip3 install --upgrade tensorflow
```

Then run

```sh
pip3 install --upgrade tf-quant-finance
```

You maybe also have to use the option ```--user```.

## TensorFlow training

If you are not familiar with TensorFlow, a good place to get started is with the
following self-study introduction to TensorFlow notebooks:

   * [Introduction to TensorFlow Part 1 - Basics](https://colab.research.google.com/github/google/tf-quant-finance/blob/master/tf_quant_finance/examples/jupyter_notebooks/Introduction_to_TensorFlow_Part_1_-_Basics.ipynb).
   * [Introduction to TensorFlow Part 2 - Debugging and Control Flow](https://colab.research.google.com/github/google/tf-quant-finance/blob/master/tf_quant_finance/examples/jupyter_notebooks/Introduction_to_TensorFlow_Part_2_-_Debugging_and_Control_Flow.ipynb).
   * [Introduction to TensorFlow Part 3 - Advanced Tensor Manipulation](https://colab.research.google.com/github/google/tf-quant-finance/blob/master/tf_quant_finance/examples/jupyter_notebooks/Introduction_to_TensorFlow_Part_3_-_Advanced_Tensor_Manipulation.ipynb).

## Development roadmap

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
      * Quadratic Local Vol model.
      * SABR model
  * ADI method for solving multi dimensional PDEs.
  * Copulas: Support for defining and sampling from copulas.
  * Model Calibration:
      * Dupire local vol calibration.
      * SABR model calibration.
  * Rate curve fitting: Hagan-West algorithm for yield curve bootstrapping and
  the Monotone Convex interpolation scheme.


## Examples
See [`tf_quant_finance/examples/`](https://github.com/google/tf-quant-finance/tree/master/tf_quant_finance/examples)
for end-to-end examples. It includes tutorial notebooks such as:

  * [American Option pricing under the Black-Scholes model](https://colab.research.google.com/github/google/tf-quant-finance/blob/master/tf_quant_finance/examples/jupyter_notebooks/American_Option_Black_Scholes.ipynb)
  * [Monte Carlo via Euler Scheme](https://colab.research.google.com/github/google/tf-quant-finance/blob/master/tf_quant_finance/examples/jupyter_notebooks/Monte_Carlo_Euler_Scheme.ipynb)
  * [Black Scholes: Price and Implied Vol](https://colab.research.google.com/github/google/tf-quant-finance/blob/master/tf_quant_finance/examples/jupyter_notebooks/Black_Scholes_Price_and_Implied_Vol.ipynb).
  * [Root search using Brent's method](https://colab.research.google.com/github/google/tf-quant-finance/blob/master/tf_quant_finance/examples/jupyter_notebooks/Root_Search.ipynb)
  * [Optimization](https://colab.research.google.com/github/google/tf-quant-finance/blob/master/tf_quant_finance/examples/jupyter_notebooks/Optimization.ipynb)

The above links will open Jupyter Notebooks in Colab.

## Contributing

We're eager to collaborate with you! See [CONTRIBUTING.md](CONTRIBUTING.md) for a guide on how to contribute. This project adheres to TensorFlow's code of conduct. By participating, you are expected to uphold this code.

## Development

This section is meant for developers who want to contribute code to the
library. If you are only interested in using the library, please follow the
instructions in the [Installation](#installation) section.

### Dependencies

This library has the following dependencies:

1.  Bazel
2.  Python 3 (Bazel uses Python 3 by default)
3.  TensorFlow version 2.1 or higher
4.  TensorFlow Probability version 0.8 or higher
5.  Numpy version 1.16 or higher
6.  Attrs

This library requires the
[Bazel](https://bazel.build/) build system. Please follow the
[Bazel installation instructions](https://docs.bazel.build/versions/master/install.html)
for your platform.


You can install TensorFlow and related dependencies using the ```pip3 install```
command:

```sh
pip3 install --upgrade tensorflow tensorflow-probability numpy attrs scipy
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

### Docker images

The official TF Quant Finance Docker images are located in the
[gcr.io](https://gcr.io/tf-quant-finance-images/tf-quant-finance) container
repository.

Images are tagged using the GitHub release version. Images contain
all development dependencies. See [Dockerfile](Dockerfile) for details.

You can pull the latest Docker image using

```sh
sudo docker pull gcr.io/tf-quant-finance-images/tf-quant-finance
```

To start a TF Quant Finance container, use the following command:

```sh
sudo docker run -it gcr.io/tf-quant-finance-images/tf-quant-finance
```

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
