# Framework for building graphs for portfolio pricing.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "framework",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/framework/core",
        "//tf_quant_finance/experimental/pricing_platform/framework/equity_instruments",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)
