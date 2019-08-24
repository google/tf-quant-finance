# Cubic interpolation methods.

licenses(["notice"])

# [internal] load python3.bzl

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
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "cubic_interpolation_test",
    size = "medium",
    srcs = ["cubic_interpolation_test.py"],
    deps = [
        ":cubic",
        # numpy dep,
        # tensorflow dep,
    ],
)
