# Utility functions.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":dataclass",
        ":tf_functions",
    ],
)

py_library(
    name = "tf_functions",
    srcs = ["tf_functions.py"],
    srcs_version = "PY3",
    deps = [
        # dataclasses dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "tf_functions_test",
    size = "small",
    srcs = ["tf_functions_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # dataclasses dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "dataclass",
    srcs = ["dataclass.py"],
    srcs_version = "PY3",
)

py_test(
    name = "dataclass_test",
    size = "small",
    srcs = ["dataclass_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)
