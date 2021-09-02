# Core tools for the pricing platform

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "core",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":business_days",
        ":currencies",
        ":curve_types",
        ":daycount_conventions",
        ":implied_volatility_type",
        ":instrument",
        ":interpolation_method",
        ":models",
        ":processed_market_data",
        ":rate_indices",
        ":types",
    ],
)

py_library(
    name = "currencies",
    srcs = ["currencies.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_library(
    name = "curve_types",
    srcs = ["curve_types.py"],
    srcs_version = "PY3",
    deps = [
        ":currencies",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_library(
    name = "business_days",
    srcs = ["business_days.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_library(
    name = "rate_indices",
    srcs = ["rate_indices.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
        # tensorflow dep,
    ],
)

py_library(
    name = "daycount_conventions",
    srcs = ["daycount_conventions.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_library(
    name = "implied_volatility_type",
    srcs = ["implied_volatility_type.py"],
    srcs_version = "PY3",
)

py_library(
    name = "interpolation_method",
    srcs = ["interpolation_method.py"],
    srcs_version = "PY3",
)

py_library(
    name = "models",
    srcs = ["models.py"],
    srcs_version = "PY3",
)

py_library(
    name = "processed_market_data",
    srcs = ["processed_market_data.py"],
    srcs_version = "PY3",
    deps = [
        ":curve_types",
        ":daycount_conventions",
        ":interpolation_method",
        ":rate_indices",
        ":types",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
        # tensorflow dep,
    ],
)

py_library(
    name = "instrument",
    srcs = ["instrument.py"],
    srcs_version = "PY3",
    deps = [
        ":curve_types",
        ":processed_market_data",
        ":types",
        # tensorflow dep,
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
    deps = [
        ":business_days",
        ":currencies",
        ":curve_types",
        ":daycount_conventions",
        ":rate_indices",
        "//tf_quant_finance/datetime",
        # tensorflow dep,
    ],
)
