# Protos for portfolio and market data representation

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "instrument_protos",
    srcs = ["__init__.py"],
    deps = [
        ":instruments_py_pb2",
    ],
)

py_proto_library(
    name = "instruments_py_pb2",
    srcs = [
        "all_instruments.proto",
        "american_equity_option.proto",
        "forward_rate_agreement.proto",
        "interest_rate_swap.proto",
        "swaption.proto",
    ],
    deps = [
        ":business_days_py_pb2",
        ":currencies_py_pb2",
        ":date_py_pb2",
        ":daycount_conventions_py_pb2",
        ":decimal_py_pb2",
        ":metadata_py_pb2",
        ":period_py_pb2",
        ":rate_indices_py_pb2",
    ],
)

py_proto_library(
    name = "daycount_conventions_py_pb2",
    srcs = ["daycount_conventions.proto"],
)

py_proto_library(
    name = "decimal_py_pb2",
    srcs = ["decimal.proto"],
)

py_proto_library(
    name = "date_py_pb2",
    srcs = ["date.proto"],
)

py_proto_library(
    name = "currencies_py_pb2",
    srcs = ["currencies.proto"],
)

py_proto_library(
    name = "metadata_py_pb2",
    srcs = ["metadata.proto"],
)

py_proto_library(
    name = "period_py_pb2",
    srcs = ["period.proto"],
)

py_proto_library(
    name = "rate_indices_py_pb2",
    srcs = ["rate_indices.proto"],
)

py_proto_library(
    name = "business_days_py_pb2",
    srcs = ["business_days.proto"],
)
