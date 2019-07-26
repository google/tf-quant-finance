# High-performance library for quantitative finance using TensorFlow

This library provides high-performance methods leveraging the hardware acceleration support and automatic differentation of TensorFlow. The initial library will provide TensorFlow support for foundational mathematical methods, mid-level methods, and specific pricing models. These will be significantly expanded over next months.

Our TensorFlow-based methods will be structured along three tiers:

1. **Foundational methods**.
Core mathematics methods - optimisation, root finders, linear algebra, random number generation, etc.

2. **Mid-level methods**.
ODE & PDE solvers, Diffusion Path Generators, Copula samplers etc.

3. **Pricing methods**.
Specific Pricing models (e.g Local Vol (LV), Stochastic Vol (SV), Stochastic Local Vol (SLV), Hull-White (HW))

## Examples
Our existing examples will be made available here soon.

## Contributing
We're eager to collaborate with you! See [CONTRIBUTING.md](CONTRIBUTING.md) for a guide on how to contribute. This project adheres to TensorFlow's code of conduct. By participating, you are expected to uphold this code.

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
