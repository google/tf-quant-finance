# Constant forward Interpolation for rate curves.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "constant_fwd",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":constant_fwd_interpolation",
        # tensorflow dep,
    ],
)

py_library(
    name = "constant_fwd_interpolation",
    srcs = ["constant_fwd_interpolation.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "constant_fwd_interpolation_test",
    size = "small",
    srcs = ["constant_fwd_interpolation_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":constant_fwd_interpolation",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
