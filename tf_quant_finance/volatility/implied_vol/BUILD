# Computation of implied volatility.

licenses(["notice"])

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
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
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
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
