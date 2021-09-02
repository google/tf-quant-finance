# Equity american option module

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "american_option",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":american_option_impl",
        ":proto_utils",
        ":utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/equity_instruments:utils",
    ],
)

py_library(
    name = "american_option_impl",
    srcs = ["american_option_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":proto_utils",
        ":utils",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:instrument",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/framework/equity_instruments:utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data:utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "american_option_impl_test",
    size = "medium",
    srcs = ["american_option_impl_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "proto_utils",
    srcs = ["proto_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/framework:utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:business_days",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:currencies",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:daycount_conventions",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/framework/equity_instruments:utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments:cashflow_streams",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments:utils",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
        # tensorflow dep,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/math",
        "//tf_quant_finance/models",
        "//tf_quant_finance/models/longstaff_schwartz",
        # tensorflow dep,
    ],
)

py_test(
    name = "utils_test",
    size = "medium",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    deps = [
        ":utils",
        # test util,
        # tensorflow dep,
    ],
)
