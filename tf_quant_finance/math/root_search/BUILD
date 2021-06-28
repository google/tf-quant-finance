# TensorFlow Quantitative Finance general math functions.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "root_search",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":brent",
        ":newton",
        ":utils",
    ],
)

py_library(
    name = "brent",
    srcs = ["brent.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "brent_test",
    size = "small",
    timeout = "moderate",
    srcs = ["brent_test.py"],
    python_version = "PY3",
    deps = [
        ":brent",
        ":utils",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "newton",
    srcs = ["newton.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "newton_test",
    size = "small",
    timeout = "moderate",
    srcs = ["newton_test.py"],
    python_version = "PY3",
    deps = [
        ":newton",
        ":utils",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
