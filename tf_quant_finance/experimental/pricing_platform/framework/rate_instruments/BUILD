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
        ":forward_rate_agreement",
        ":interest_rate_swap",
        ":utils",
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
    name = "swap_utils",
    srcs = ["swap_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":coupon_specs",
        ":utils",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:business_days",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:currencies",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:daycount_conventions",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:rate_indices",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
        # tensorflow dep,
    ],
)

py_library(
    name = "coupon_specs",
    srcs = ["coupon_specs.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        # dataclasses dep,
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
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data:utils",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
        # tensorflow dep,
    ],
)

py_library(
    name = "interest_rate_swap",
    srcs = ["interest_rate_swap.py"],
    srcs_version = "PY3",
    deps = [
        ":cashflow_streams",
        ":coupon_specs",
        ":swap_utils",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:instrument",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:rate_indices",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data:utils",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
        "//tf_quant_finance/math",
        # dataclasses dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "interest_rate_swap_test",
    size = "medium",
    srcs = ["interest_rate_swap_test.py"],
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
    name = "forward_rate_agreement",
    srcs = ["forward_rate_agreement.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:business_days",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:currencies",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:daycount_conventions",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:instrument",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:rate_indices",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data:utils",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
        # dataclasses dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "forward_rate_agreement_test",
    srcs = ["forward_rate_agreement_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        "//tf_quant_finance",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:date_py_pb2",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:decimal_py_pb2",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:instruments_py_pb2",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
