# Experimental modules.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "pde_v2",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fd_solvers",
    ],
)

py_library(
    name = "fd_solvers",
    srcs = ["fd_solvers.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)
