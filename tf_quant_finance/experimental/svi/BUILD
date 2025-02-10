# SVI model and its flavors.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "svi",
    srcs = ["__init__.py"],
    deps = [
        ":calibration",
        ":parameterizations",
    ],
)

py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    deps = [
        ":parameterizations",
        "//tf_quant_finance/math",
        "//tf_quant_finance/math/optimizer",
        # tensorflow dep,
    ],
)

py_test(
    name = "calibration_test",
    size = "small",
    timeout = "long",
    srcs = ["calibration_test.py"],
    deps = [
        ":calibration",
        ":parameterizations",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_library(
    name = "parameterizations",
    srcs = ["parameterizations.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "parameterizations_test",
    size = "small",
    timeout = "moderate",
    srcs = ["parameterizations_test.py"],
    deps = [
        ":parameterizations",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
