# Tools to build Diffusion Models

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "models",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":euler_sampling",
        ":generic_ito_process",
        ":ito_process",
        ":joined_ito_process",
        ":milstein_sampling",
        ":realized_volatility",
        ":valuation_method",
        "//tf_quant_finance/models/cir",
        "//tf_quant_finance/models/geometric_brownian_motion",
        "//tf_quant_finance/models/heston",
        "//tf_quant_finance/models/hjm",
        "//tf_quant_finance/models/hull_white",
        "//tf_quant_finance/models/longstaff_schwartz",
        "//tf_quant_finance/models/sabr",
    ],
)

py_library(
    name = "valuation_method",
    srcs = ["valuation_method.py"],
    strict_deps = False,
)

py_library(
    name = "ito_process",
    srcs = ["ito_process.py"],
    strict_deps = False,
    deps = [],
)

py_library(
    name = "generic_ito_process",
    srcs = ["generic_ito_process.py"],
    strict_deps = False,
    deps = [
        ":euler_sampling",
        ":ito_process",
        "//tf_quant_finance/math/pde",
        # tensorflow dep,
    ],
)

py_test(
    name = "generic_ito_process_test",
    size = "small",
    timeout = "moderate",
    srcs = ["generic_ito_process_test.py"],
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)

py_library(
    name = "euler_sampling",
    srcs = ["euler_sampling.py"],
    strict_deps = False,
    deps = [
        ":utils",
        "//tf_quant_finance/math:custom_loops",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "euler_sampling_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["euler_sampling_test.py"],
    shard_count = 20,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "milstein_sampling",
    srcs = ["milstein_sampling.py"],
    strict_deps = False,
    deps = [
        ":utils",
        "//tf_quant_finance/math:custom_loops",
        "//tf_quant_finance/math:gradient",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "milstein_sampling_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["milstein_sampling_test.py"],
    shard_count = 6,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "joined_ito_process",
    srcs = ["joined_ito_process.py"],
    strict_deps = False,
    deps = [
        ":euler_sampling",
        ":generic_ito_process",
        ":ito_process",
        ":utils",
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/math/random_ops",
        # tensorflow dep,
    ],
)

py_test(
    name = "joined_ito_process_test",
    size = "medium",
    timeout = "long",
    srcs = ["joined_ito_process_test.py"],
    shard_count = 3,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/math/random_ops",
        # tensorflow dep,
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    shard_count = 3,
    deps = [
        ":utils",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "realized_volatility",
    srcs = ["realized_volatility.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/math:diff_ops",
        # tensorflow dep,
    ],
)

py_test(
    name = "realized_volatility_test",
    size = "small",
    srcs = ["realized_volatility_test.py"],
    deps = [
        "//tf_quant_finance",
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
    data = [
        "//tf_quant_finance/models/legacy:docs",
    ],
)
