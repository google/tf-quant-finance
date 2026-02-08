# TensorFlow Quantitative Finance general math functions.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "math",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":diff_ops",
        ":gradient",
        ":jacobian",
        ":pad",
        ":piecewise",
        ":segment_ops",
        "//tf_quant_finance/math/integration",
        "//tf_quant_finance/math/interpolation",
        "//tf_quant_finance/math/optimizer",
        "//tf_quant_finance/math/pde",
        "//tf_quant_finance/math/qmc",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/math/root_search",
    ],
)

py_library(
    name = "diff_ops",
    srcs = ["diff_ops.py"],
    strict_deps = False,
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "diff_ops_test",
    size = "small",
    srcs = ["diff_ops_test.py"],
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "gradient",
    srcs = ["gradient.py"],
    strict_deps = False,
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "gradient_test",
    size = "small",
    srcs = ["gradient_test.py"],
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "jacobian",
    srcs = ["jacobian.py"],
    strict_deps = False,
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "jacobian_test",
    size = "small",
    timeout = "moderate",
    srcs = ["jacobian_test.py"],
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "segment_ops",
    srcs = ["segment_ops.py"],
    strict_deps = False,
    deps = [
        ":diff_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "segment_ops_test",
    size = "small",
    srcs = ["segment_ops_test.py"],
    deps = [
        ":segment_ops",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "piecewise",
    srcs = ["piecewise.py"],
    strict_deps = False,
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "piecewise_test",
    size = "medium",
    srcs = ["piecewise_test.py"],
    shard_count = 5,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "custom_loops",
    srcs = ["custom_loops.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "custom_loops_test",
    size = "small",
    timeout = "moderate",
    srcs = ["custom_loops_test.py"],
    shard_count = 12,
    deps = [
        ":custom_loops",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)

py_library(
    name = "pad",
    srcs = ["pad.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/datetime:date_tensor",
        # tensorflow dep,
    ],
)

py_test(
    name = "pad_test",
    size = "small",
    srcs = ["pad_test.py"],
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
    data = [
        "//tf_quant_finance/math/interpolation:docs",
        "//tf_quant_finance/math/optimizer:docs",
        "//tf_quant_finance/math/pde:docs",
        "//tf_quant_finance/math/random_ops:docs",
    ],
)
