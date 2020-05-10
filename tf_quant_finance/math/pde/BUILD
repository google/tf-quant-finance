# PDE solver methods.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "pde",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":boundary_conditions",
        ":fd_solvers",
        ":grids",
    ],
)

py_library(
    name = "fd_solvers",
    srcs = ["fd_solvers.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/math/pde/steppers",
        # tensorflow dep,
    ],
)

py_library(
    name = "boundary_conditions",
    srcs = ["boundary_conditions.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "grids",
    srcs = ["grids.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
        "pde_solvers.pdf",
    ],
)
