# RQMC (random quasi-Monte-Carlo) support

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "qmc",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":digital_net",
        ":lattice_rule",
        ":sobol",
        # tensorflow dep,
    ],
)

py_library(
    name = "digital_net",
    srcs = ["digital_net.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "digital_net_test",
    srcs = ["digital_net_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "lattice_rule",
    srcs = ["lattice_rule.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "lattice_rule_test",
    srcs = ["lattice_rule_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "sobol",
    srcs = ["sobol.py"],
    srcs_version = "PY3",
    deps = [
        ":digital_net",
        ":utils",
        "//tf_quant_finance/math/random_ops/sobol",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "sobol_test",
    srcs = ["sobol_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)
