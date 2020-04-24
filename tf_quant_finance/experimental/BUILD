# Experimental modules.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":io",
        "//tf_quant_finance/experimental/dates",
        "//tf_quant_finance/experimental/finite_difference",
        "//tf_quant_finance/experimental/instruments",
        "//tf_quant_finance/experimental/lsm_algorithm",
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
    data = [
        "//tf_quant_finance/experimental/finite_difference:docs",
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "io_test",
    size = "small",
    srcs = ["io_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
