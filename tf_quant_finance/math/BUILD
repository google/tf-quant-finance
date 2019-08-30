# TensorFlow Quantitative Finance general math functions.

licenses(["notice"])

# [internal] load python3.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "math",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":root_search",
        "//tf_quant_finance/math/interpolation",
        "//tf_quant_finance/math/pde",
        "//tf_quant_finance/math/random",
    ],
)

py_library(
    name = "root_search",
    srcs = ["root_search.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "root_search_test",
    size = "small",
    srcs = ["root_search_test.py"],
    deps = [
        ":root_search",
        # numpy dep,
        # tensorflow dep,
    ],
)
