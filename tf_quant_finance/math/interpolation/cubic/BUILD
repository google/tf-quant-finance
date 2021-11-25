# Cubic interpolation methods.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "cubic",
    srcs = ["__init__.py"],
    deps = [
        ":cubic_interpolation",
        # tensorflow dep,
    ],
)

py_library(
    name = "cubic_interpolation",
    srcs = ["cubic_interpolation.py"],
    deps = [
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "cubic_interpolation_test",
    size = "medium",
    srcs = ["cubic_interpolation_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)
