# PDE solver methods.

# Placeholder: load py_library

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "pde",
    srcs = ["__init__.py"],
    deps = [
        ":boundary_conditions",
        ":fd_solvers",
        ":grids",
    ],
)

py_library(
    name = "fd_solvers",
    srcs = ["fd_solvers.py"],
    deps = [
        "//tf_quant_finance/math/pde/steppers",
        # tensorflow dep,
    ],
)

py_library(
    name = "boundary_conditions",
    srcs = ["boundary_conditions.py"],
    deps = [],
)

py_library(
    name = "grids",
    srcs = ["grids.py"],
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
