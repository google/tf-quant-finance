# Interpolation methods for one and two dimensions.

# Placeholder: load py_library

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "interpolation",
    srcs = ["__init__.py"],
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
