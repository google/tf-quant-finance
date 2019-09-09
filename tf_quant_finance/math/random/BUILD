# Ops related to random or quasi random number generation.

licenses(["notice"])

# [internal] load python3.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "random",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":multivariate_normal",
        ":stateless",
        "//tf_quant_finance/math/random/halton",
        "//tf_quant_finance/math/random/sobol",
    ],
)

py_library(
    name = "multivariate_normal",
    srcs = ["multivariate_normal.py"],
    srcs_version = "PY2AND3",
    deps = [
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
    deps = [
        ":stateless",
        # numpy dep,
        # tensorflow dep,
    ],
)
