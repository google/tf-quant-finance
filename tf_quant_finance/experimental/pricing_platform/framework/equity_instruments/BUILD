# Equity instruments module

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "equity_instruments",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/equity_instruments/american_option",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)
