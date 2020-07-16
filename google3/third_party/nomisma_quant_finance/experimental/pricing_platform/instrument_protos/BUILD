# Protos for portfolio and market data representation

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "instruments_proto",
    srcs = [
        "forward_rate_agreement.proto",
        "interest_rate_swap.proto",
    ],
    cc_api_version = 2,
    deps = [
        ":business_days_proto",
        ":currencies_proto",
        ":date_proto",
        ":day_count_conventions_proto",
        ":decimal_proto",
        ":metadata_proto",
        ":period_proto",
        ":rate_indices_proto",
    ],
)

py_proto_library(
    name = "instruments_py_pb2",
    api_version = 2,
    deps = [":instruments_proto"],
)

proto_library(
    name = "day_count_conventions_proto",
    srcs = ["day_count_conventions.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "day_count_conventions_py_pb2",
    api_version = 2,
    deps = [":day_count_conventions_proto"],
)

proto_library(
    name = "decimal_proto",
    srcs = ["decimal.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "decimal_py_pb2",
    api_version = 2,
    deps = [":decimal_proto"],
)

proto_library(
    name = "date_proto",
    srcs = ["date.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "date_py_pb2",
    api_version = 2,
    deps = [":date_proto"],
)

proto_library(
    name = "currencies_proto",
    srcs = ["currencies.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "currencies_py_pb2",
    api_version = 2,
    deps = [":currencies_proto"],
)

proto_library(
    name = "metadata_proto",
    srcs = ["metadata.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "metadata_py_pb2",
    api_version = 2,
    deps = [":metadata_proto"],
)

proto_library(
    name = "period_proto",
    srcs = ["period.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "period_py_pb2",
    api_version = 2,
    deps = [":period_proto"],
)

proto_library(
    name = "rate_indices_proto",
    srcs = ["rate_indices.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "rate_indices_py_pb2",
    api_version = 2,
    deps = [":rate_indices_proto"],
)

proto_library(
    name = "business_days_proto",
    srcs = ["business_days.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "business_days_py_pb2",
    api_version = 2,
    deps = [":business_days_proto"],
)
