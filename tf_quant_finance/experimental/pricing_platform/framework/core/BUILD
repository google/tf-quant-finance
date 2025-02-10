# Core tools for the pricing platform

# Placeholder: load py_library

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "core",
    srcs = ["__init__.py"],
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
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_library(
    name = "curve_types",
    srcs = ["curve_types.py"],
    deps = [
        ":currencies",
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_library(
    name = "business_days",
    srcs = ["business_days.py"],
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_library(
    name = "rate_indices",
    srcs = ["rate_indices.py"],
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
        # tensorflow dep,
    ],
)

py_library(
    name = "daycount_conventions",
    srcs = ["daycount_conventions.py"],
    deps = [
        "//tf_quant_finance/experimental/pricing_platform/instrument_protos",
    ],
)

py_library(
    name = "implied_volatility_type",
    srcs = ["implied_volatility_type.py"],
)

py_library(
    name = "interpolation_method",
    srcs = ["interpolation_method.py"],
)

py_library(
    name = "models",
    srcs = ["models.py"],
)

py_library(
    name = "processed_market_data",
    srcs = ["processed_market_data.py"],
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
