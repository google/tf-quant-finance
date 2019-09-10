# Interpolation methods for one and two dimensions.

licenses(["notice"])

# [internal] load python3.bzl

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
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "linear_interpolation_test",
    size = "small",
    srcs = ["linear_interpolation_test.py"],
    #shard_count = 3,
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
    ],
)
