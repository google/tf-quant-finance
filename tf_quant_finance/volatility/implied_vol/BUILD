# Computation of implied volatility.

licenses(["notice"])

# [internal] load python3.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "implied_vol",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":newton_root",
        ":polya_approx",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "polya_approx",
    srcs = ["polya_approx.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "polya_approx_test",
    srcs = ["polya_approx_test.py"],
    deps = [
        ":polya_approx",
        "//tf_quant_finance/volatility:black_scholes",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_library(
    name = "newton_root",
    srcs = ["newton_root.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":polya_approx",
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "newton_root_test",
    size = "medium",
    srcs = ["newton_root_test.py"],
    shard_count = 4,
    deps = [
        ":newton_root",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
