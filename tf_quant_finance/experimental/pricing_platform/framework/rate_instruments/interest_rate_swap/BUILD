# Interest Rate Swap instrument

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "interest_rate_swap",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":interest_rate_swap_impl",
    ],
)

py_library(
    name = "interest_rate_swap_impl",
    srcs = ["interest_rate_swap_impl.py"],
    strict_deps = False,
    deps = [
        ":proto_utils",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:instrument",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data:utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments:cashflow_streams",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments:coupon_specs",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
        "//tf_quant_finance/math",
        # tensorflow dep,
    ],
)

py_test(
    name = "interest_rate_swap_test",
    size = "medium",
    srcs = ["interest_rate_swap_test.py"],
    shard_count = 6,
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "proto_utils",
    srcs = ["proto_utils.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/framework:utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:business_days",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:currencies",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:daycount_conventions",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:rate_indices",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data:utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments:coupon_specs",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments:utils",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_test(
    name = "proto_utils_test",
    size = "small",
    srcs = ["proto_utils_test.py"],
    shard_count = 2,
    strict_deps = False,
    deps = [
        ":proto_utils",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
