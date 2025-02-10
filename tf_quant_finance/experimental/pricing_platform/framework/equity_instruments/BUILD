# Equity instruments module

# Placeholder: load py_library

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "equity_instruments",
    srcs = ["__init__.py"],
    deps = [
        ":utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/equity_instruments/american_option",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)
