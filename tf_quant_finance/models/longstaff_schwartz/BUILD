# LSM algorithm methods.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "longstaff_schwartz",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":lsm",
        ":payoff_utils",
    ],
)

py_library(
    name = "lsm",
    srcs = ["lsm.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "lsm_test",
    size = "small",
    timeout = "short",
    srcs = ["lsm_test.py"],
    shard_count = 4,
    strict_deps = False,
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
    name = "payoff_utils",
    srcs = ["payoff_utils.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "payoff_utils_test",
    size = "small",
    srcs = ["payoff_utils_test.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
