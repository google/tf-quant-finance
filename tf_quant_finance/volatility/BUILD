# Tools for fitting volatility surfaces and pricing of vanilla options.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "volatility",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":black_scholes",
        "//tf_quant_finance/volatility/implied_vol",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "black_scholes",
    srcs = ["black_scholes.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "black_scholes_test",
    srcs = ["black_scholes_test.py"],
    python_version = "PY3",
    deps = [
        ":black_scholes",
        # numpy dep,
        # tensorflow dep,
    ],
)
