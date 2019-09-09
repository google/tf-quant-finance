# TensorFlow Quantitative Finance general math functions.

licenses(["notice"])

# [internal] load python3.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "math",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":diff",
        ":gradient",
        ":piecewise",
        ":root_search",
        ":segment_ops",
        "//tf_quant_finance/math/interpolation",
        "//tf_quant_finance/math/optimizer",
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

py_library(
    name = "diff",
    srcs = ["diff.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "diff_test",
    size = "small",
    srcs = ["diff_test.py"],
    deps = [
        ":diff",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "gradient",
    srcs = ["gradient.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "gradient_test",
    size = "small",
    srcs = ["gradient_test.py"],
    deps = [
        ":gradient",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "segment_ops",
    srcs = ["segment_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":diff",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "segment_ops_test",
    size = "small",
    srcs = ["segment_ops_test.py"],
    deps = [
        ":segment_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "piecewise",
    srcs = ["piecewise.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "piecewise_test",
    size = "small",
    srcs = ["piecewise_test.py"],
    deps = [
        ":piecewise",
        # numpy dep,
        # tensorflow dep,
    ],
)
