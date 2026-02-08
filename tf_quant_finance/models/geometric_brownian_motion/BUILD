# Geometric Brownian Motion

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "geometric_brownian_motion",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":multivariate_geometric_brownian_motion",
        ":univariate_geometric_brownian_motion",
    ],
)

py_library(
    name = "geometric_brownian_motion_test_utils",
    srcs = ["geometric_brownian_motion_test_utils.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/models:utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "univariate_geometric_brownian_motion",
    srcs = ["univariate_geometric_brownian_motion.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/math/pde",
        "//tf_quant_finance/models:ito_process",
        "//tf_quant_finance/models:utils",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "multivariate_geometric_brownian_motion",
    srcs = ["multivariate_geometric_brownian_motion.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/math/pde",
        "//tf_quant_finance/models:ito_process",
        "//tf_quant_finance/models:utils",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "geometric_brownian_motion_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["geometric_brownian_motion_test.py"],
    shard_count = 8,
    deps = [
        ":geometric_brownian_motion",
        ":geometric_brownian_motion_test_utils",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)

py_test(
    name = "geometric_brownian_motion_test_utils_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["geometric_brownian_motion_test_utils_test.py"],
    shard_count = 8,
    deps = [
        ":geometric_brownian_motion",
        ":geometric_brownian_motion_test_utils",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
