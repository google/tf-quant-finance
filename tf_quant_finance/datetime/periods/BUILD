# TODO(cyrilchimisov): describe this package.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "periods",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":period_tensors",
        # tensorflow dep,
    ],
)

py_library(
    name = "period_tensors",
    srcs = ["period_tensors.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/datetime:constants",
        "//tf_quant_finance/datetime:tensor_wrapper",
        # tensorflow dep,
    ],
)
