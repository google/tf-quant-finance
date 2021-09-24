# Rate instruments module

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "rate_instruments",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":cashflow_streams",
        ":coupon_specs",
        ":utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments/forward_rate_agreement",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments/interest_rate_swap",
        "//tf_quant_finance/experimental/pricing_platform/framework/rate_instruments/swaption",
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

py_library(
    name = "coupon_specs",
    srcs = ["coupon_specs.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        # tensorflow dep,
    ],
)

py_library(
    name = "cashflow_streams",
    srcs = ["cashflow_streams.py"],
    srcs_version = "PY3",
    deps = [
        ":coupon_specs",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data:rate_curve",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data:utils",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
        "//tf_quant_finance/math",
        # tensorflow dep,
    ],
)

py_test(
    name = "cashflow_streams_test",
    size = "medium",
    srcs = ["cashflow_streams_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # tensorflow dep,
    ],
)
