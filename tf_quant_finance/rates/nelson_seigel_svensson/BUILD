# Nelson Seigel Svensson Interpolation for rate curves.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "nelson_seigel_svensson",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":nelson_seigel_svensson_interpolation",
        # tensorflow dep,
    ],
)

py_library(
    name = "nelson_seigel_svensson_interpolation",
    srcs = ["nelson_seigel_svensson_interpolation.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "nelson_seigel_svensson_interpolation_test",
    size = "small",
    srcs = ["nelson_seigel_svensson_interpolation_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":nelson_seigel_svensson_interpolation",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
