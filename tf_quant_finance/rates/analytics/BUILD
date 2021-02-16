# Tools for handling swap construction and cashflow manipulation

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "analytics",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":cashflows",
        ":forwards",
        ":swap",
    ],
)

py_library(
    name = "forwards",
    srcs = ["forwards.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/math:segment_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "forwards_test",
    size = "small",
    srcs = ["forwards_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "cashflows",
    srcs = ["cashflows.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "cashflows_test",
    size = "small",
    srcs = ["cashflows_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "swap",
    srcs = ["swap.py"],
    srcs_version = "PY3",
    deps = [
        ":cashflows",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "swap_test",
    size = "small",
    srcs = ["swap_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
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
