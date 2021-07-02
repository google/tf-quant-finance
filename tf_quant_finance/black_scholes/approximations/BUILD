# Approximation methods in Black-Scholes market.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "approximations",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":american_option",
        # tensorflow dep,
    ],
)

py_library(
    name = "american_option",
    srcs = ["american_option.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/black_scholes:implied_vol_newton_root",
        "//tf_quant_finance/black_scholes:vanilla_prices",
        "//tf_quant_finance/math",
        "//tf_quant_finance/math/root_search:newton",
        "//tf_quant_finance/types",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "american_option_test",
    srcs = ["american_option_test.py"],
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
