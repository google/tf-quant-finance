# HJM model and its flavors.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "hjm",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":quasi_gaussian_hjm",
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
    name = "zero_coupon_bond_option_util",
    srcs = ["zero_coupon_bond_option_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/models:utils",
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
    timeout = "long",
    srcs = ["zero_coupon_bond_option_test.py"],
    python_version = "PY3",
    shard_count = 7,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)
