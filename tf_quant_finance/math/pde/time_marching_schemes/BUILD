# PDE time marching schemes.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "time_marching_schemes",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":pde_time_marching_scheme",
        ":pde_time_marching_schemes",
    ],
)

py_library(
    name = "pde_time_marching_schemes",
    srcs = [
        "pde_time_marching_schemes.py",
    ],
    deps = [
        ":pde_time_marching_scheme",
        "//tf_quant_finance/math/pde/internal",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "pde_time_marching_scheme",
    srcs = [
        "pde_time_marching_scheme.py",
    ],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)
