# Framework for building graphs for portfolio pricing.

# Placeholder: load py_library

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "framework",
    srcs = ["__init__.py"],
    strict_deps = False,
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
    strict_deps = False,
)
