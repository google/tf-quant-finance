# Utility functions.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":dataclass",
        ":shape_utils",
        ":tf_functions",
    ],
)

py_library(
    name = "tf_functions",
    srcs = ["tf_functions.py"],
    strict_deps = False,
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "tf_functions_test",
    size = "small",
    srcs = ["tf_functions_test.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "dataclass",
    srcs = ["dataclass.py"],
    strict_deps = False,
)

py_test(
    name = "dataclass_test",
    size = "small",
    srcs = ["dataclass_test.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "shape_utils",
    srcs = ["shape_utils.py"],
    strict_deps = False,
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "shape_utils_test",
    size = "small",
    srcs = ["shape_utils_test.py"],
    shard_count = 2,
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)
