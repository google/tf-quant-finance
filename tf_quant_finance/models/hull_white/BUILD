# Hull White model
package(default_visibility = ["//tf_quant_finance:__subpackages__"])

licenses(["notice"])

py_library(
    name = "hull_white",
    srcs = ["__init__.py"],
    deps = [
        ":cap_floor",
        ":one_factor",
        ":swaption",
        ":vector_hull_white",
        ":zero_coupon_bond_option",
    ],
)

py_library(
    name = "vector_hull_white",
    srcs = ["vector_hull_white.py"],
    deps = [
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/models:generic_ito_process",
        "//tf_quant_finance/models:utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "one_factor",
    srcs = ["one_factor.py"],
    deps = [
        ":vector_hull_white",
    ],
)

py_test(
    name = "hull_white_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["hull_white_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "zero_coupon_bond_option",
    srcs = ["zero_coupon_bond_option.py"],
    deps = [
        ":vector_hull_white",
    ],
)

py_test(
    name = "zero_coupon_bond_option_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["zero_coupon_bond_option_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)

py_library(
    name = "cap_floor",
    srcs = ["cap_floor.py"],
    deps = [
        ":vector_hull_white",
        ":zero_coupon_bond_option",
    ],
)

py_test(
    name = "cap_floor_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["cap_floor_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)

py_library(
    name = "swaption",
    srcs = ["swaption.py"],
    deps = [
        ":vector_hull_white",
        ":zero_coupon_bond_option",
    ],
)

py_test(
    name = "swaption_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["swaption_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)
