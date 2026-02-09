# Implementation of the Hagan-West algorithm for fitting rate curves.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "hagan_west",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":bond_curve",
        ":monotone_convex",
    ],
)

py_library(
    name = "monotone_convex",
    srcs = ["monotone_convex.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/math:diff_ops",
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/rates/analytics:forwards",
        "//tf_quant_finance/types",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "monotone_convex_test",
    size = "small",
    srcs = ["monotone_convex_test.py"],
    shard_count = 5,
    strict_deps = False,
    deps = [
        ":monotone_convex",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "bond_curve",
    srcs = ["bond_curve.py"],
    strict_deps = False,
    deps = [
        ":monotone_convex",
        "//tf_quant_finance/math:pad",
        "//tf_quant_finance/rates/analytics:cashflows",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "bond_curve_test",
    size = "medium",
    srcs = ["bond_curve_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":bond_curve",
        ":monotone_convex",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
