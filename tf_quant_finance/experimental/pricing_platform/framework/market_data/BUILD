# Market data interface implementation.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "market_data",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":market_data_config",
        ":market_data_impl",
        ":rate_curve",
        ":utils",
        ":volatility_surface",
    ],
)

py_library(
    name = "market_data_impl",
    srcs = ["market_data_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":market_data_config",
        ":volatility_surface",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:currencies",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
        # tensorflow dep,
    ],
)

py_test(
    name = "market_data_test",
    size = "small",
    srcs = ["market_data_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
    ],
)

py_library(
    name = "market_data_config",
    srcs = ["market_data_config.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/framework/core:currencies",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:daycount_conventions",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:interpolation_method",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:business_days",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:daycount_conventions",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:rate_indices",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
        # tensorflow dep,
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
    ],
)

py_library(
    name = "rate_curve",
    srcs = ["rate_curve.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:daycount_conventions",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:interpolation_method",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
        "//tf_quant_finance/math",
        "//tf_quant_finance/rates",
        # tensorflow dep,
    ],
)

py_test(
    name = "rate_curve_test",
    size = "small",
    srcs = ["rate_curve_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        "//tf_quant_finance",
        # test util,
    ],
)

py_library(
    name = "volatility_surface",
    srcs = ["volatility_surface.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:curve_types",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:daycount_conventions",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:implied_volatility_type",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:processed_market_data",
        "//tf_quant_finance/experimental/pricing_platform/framework/core:types",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos:period_py_pb2",
        "//tf_quant_finance/math",
        "//tf_quant_finance/rates",
        # tensorflow dep,
    ],
)

py_test(
    name = "volatility_surface_test",
    size = "small",
    srcs = ["volatility_surface_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        "//tf_quant_finance",
        # test util,
    ],
)
