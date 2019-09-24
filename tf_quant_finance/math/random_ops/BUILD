# Ops related to random or quasi random number generation.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "random_ops",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":multivariate_normal",
        ":stateless",
        "//tf_quant_finance/math/random_ops/halton",
        "//tf_quant_finance/math/random_ops/sobol",
    ],
)

py_library(
    name = "multivariate_normal",
    srcs = ["multivariate_normal.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tf_quant_finance/math/random_ops/halton",
        "//tf_quant_finance/math/random_ops/sobol",
        # enum dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_library(
    name = "stateless",
    srcs = [
        "stateless.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "multivariate_normal_test",
    size = "small",
    srcs = ["multivariate_normal_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "stateless_test",
    size = "small",
    srcs = ["stateless_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
    ],
)
