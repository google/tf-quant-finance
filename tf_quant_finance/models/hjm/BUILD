# HJM model and its flavors.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "hjm",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":calibration",
        ":cap_floor",
        ":gaussian_hjm",
        ":quasi_gaussian_hjm",
        ":swaption_pricing",
        ":zero_coupon_bond_option",
    ],
)

py_library(
    name = "quasi_gaussian_hjm",
    srcs = ["quasi_gaussian_hjm.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/models:generic_ito_process",
        "//tf_quant_finance/models:utils",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "quasi_gaussian_hjm_test",
    size = "medium",
    timeout = "long",
    srcs = ["quasi_gaussian_hjm_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        ":quasi_gaussian_hjm",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "gaussian_hjm",
    srcs = ["gaussian_hjm.py"],
    srcs_version = "PY3",
    deps = [
        "quasi_gaussian_hjm",
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/models:generic_ito_process",
        "//tf_quant_finance/models:utils",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "gaussian_hjm_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["gaussian_hjm_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        ":gaussian_hjm",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "zero_coupon_bond_option_util",
    srcs = ["zero_coupon_bond_option_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/models:utils",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "zero_coupon_bond_option",
    srcs = ["zero_coupon_bond_option.py"],
    srcs_version = "PY3",
    deps = [
        ":quasi_gaussian_hjm",
        ":zero_coupon_bond_option_util",
        # tensorflow dep,
    ],
)

py_test(
    name = "zero_coupon_bond_option_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["zero_coupon_bond_option_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)

py_library(
    name = "cap_floor",
    srcs = ["cap_floor.py"],
    deps = [
        ":zero_coupon_bond_option",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "cap_floor_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["cap_floor_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "swaption_util",
    srcs = ["swaption_util.py"],
    deps = [
        "//tf_quant_finance/models:utils",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_library(
    name = "swaption_pricing",
    srcs = ["swaption_pricing.py"],
    deps = [
        ":gaussian_hjm",
        ":quasi_gaussian_hjm",
        ":swaption_util",
        "//tf_quant_finance/math",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "swaption_pricing_test",
    timeout = "long",
    srcs = ["swaption_pricing_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    deps = [
        ":quasi_gaussian_hjm",
        ":swaption_util",
        "//tf_quant_finance/black_scholes",
        "//tf_quant_finance/math:gradient",
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/math/optimizer",
        "//tf_quant_finance/math/root_search",
        "//tf_quant_finance/rates/analytics:swap",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "calibration_test",
    timeout = "long",
    srcs = ["calibration_test.py"],
    python_version = "PY3",
    shard_count = 12,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "calibration_xla_test",
    timeout = "long",
    srcs = ["calibration_xla_test.py"],
    python_version = "PY3",
    shard_count = 1,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)
