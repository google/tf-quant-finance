# Interpolation methods for one and two dimensions.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "interpolation",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/math/interpolation/cubic",
        "//tf_quant_finance/math/interpolation/linear",
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

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    deps = [
        ":utils",
        # test util,
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
