# Experimental modules.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "pde_v2",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":boundary_conditions",
        ":fd_solvers",
        "//tf_quant_finance/experimental/pde_v2/grids",
    ],
)

py_library(
    name = "fd_solvers",
    srcs = ["fd_solvers.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tf_quant_finance/experimental/pde_v2/fd_backward_schemes",
        # tensorflow dep,
    ],
)

py_library(
    name = "boundary_conditions",
    srcs = ["boundary_conditions.py"],
    srcs_version = "PY2AND3",
    deps = [],
)
