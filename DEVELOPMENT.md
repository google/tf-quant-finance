# Development Notes

The package requires TensorFlow and TensorFlow Probability for numerical model imports and Bazel for the full repository test graph. The lightweight compatibility checks can run without those dependencies:

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q tf_quant_finance tests
```

The package initializer no longer depends on the removed `distutils` module and accepts release, patch, and prerelease TensorFlow version strings through a small dependency-free parser.

For full validation, install the TensorFlow and TensorFlow Probability versions described in `README.md`, install Bazel, and run the repository's Bazel test targets. The Docker build remains the reproducible environment definition for that path.
