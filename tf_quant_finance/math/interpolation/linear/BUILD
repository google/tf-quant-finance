# Interpolation methods for one and two dimensions.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "linear",
    srcs = ["__init__.py"],
    deps = [
        ":linear_interpolation",
        # tensorflow dep,
    ],
)

py_library(
    name = "linear_interpolation",
    srcs = ["linear_interpolation.py"],
    deps = [
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "linear_interpolation_test",
    size = "small",
    srcs = ["linear_interpolation_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
