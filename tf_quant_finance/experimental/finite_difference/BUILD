# Finite difference methods.

# Placeholder: load py_library

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "finite_difference",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":methods",
    ],
)

py_library(
    name = "methods",
    srcs = ["methods.py"],
    strict_deps = False,
    deps = [
        # autograd dep,
        # pandas dep,
        # sobol_seq dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
