# RQMC (random quasi-Monte-Carlo) support

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "qmc",
    srcs = ["__init__.py"],
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
    deps = [
        ":utils",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "digital_net_test",
    srcs = ["digital_net_test.py"],
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "lattice_rule",
    srcs = ["lattice_rule.py"],
    deps = [
        ":utils",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "lattice_rule_test",
    srcs = ["lattice_rule_test.py"],
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "sobol",
    srcs = ["sobol.py"],
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
    deps = [
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)
