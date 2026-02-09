# Constant forward Interpolation for rate curves.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "constant_fwd",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":constant_fwd_interpolation",
        # tensorflow dep,
    ],
)

py_library(
    name = "constant_fwd_interpolation",
    srcs = ["constant_fwd_interpolation.py"],
    strict_deps = False,
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "constant_fwd_interpolation_test",
    size = "small",
    srcs = ["constant_fwd_interpolation_test.py"],
    shard_count = 2,
    strict_deps = False,
    deps = [
        ":constant_fwd_interpolation",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
