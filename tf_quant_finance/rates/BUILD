# Tools for handling rate curves.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "rates",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cashflows",
        ":forwards",
        "//tf_quant_finance/rates/hagan_west",
    ],
)

py_library(
    name = "forwards",
    srcs = ["forwards.py"],
    srcs_version = "PY2AND3",
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
    deps = [
        ":forwards",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "cashflows",
    srcs = ["cashflows.py"],
    srcs_version = "PY2AND3",
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
    deps = [
        ":cashflows",
        # numpy dep,
        # tensorflow dep,
    ],
)
