# Cubic interpolation methods.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

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
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "cubic_interpolation_test",
    size = "medium",
    srcs = ["cubic_interpolation_test.py"],
    shard_count = 2,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)
