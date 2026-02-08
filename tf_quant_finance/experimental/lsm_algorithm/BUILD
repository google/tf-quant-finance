# LSM algorithm methods.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "lsm_algorithm",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":lsm",
        ":lsm_v2",
        ":payoff",
    ],
)

py_library(
    name = "lsm",
    srcs = ["lsm.py"],
    strict_deps = False,
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "lsm_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["lsm_test.py"],
    shard_count = 3,
    deps = [
        ":lsm",
        ":payoff",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "lsm_v2",
    srcs = ["lsm_v2.py"],
    strict_deps = False,
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "lsm_v2_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["lsm_v2_test.py"],
    shard_count = 10,
    deps = [
        ":payoff",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)

py_library(
    name = "payoff",
    srcs = ["payoff.py"],
    strict_deps = False,
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "payoff_test",
    size = "small",
    srcs = ["payoff_test.py"],
    deps = [
        ":payoff",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
