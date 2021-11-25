# Interpolation methods for one and two dimensions.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "interpolation",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/math/interpolation/cubic",
        "//tf_quant_finance/math/interpolation/interpolation_2d",
        "//tf_quant_finance/math/interpolation/linear",
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
