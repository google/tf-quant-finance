# Tools for fitting volatility surfaces and pricing of vanilla options.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "black_scholes",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":asian_prices",
        ":brownian_bridge",
        ":crr_binomial_tree",
        ":implied_vol_approximation",
        ":implied_vol_lib",
        ":implied_vol_newton_root",
        ":vanilla_prices",
        ":variance_swaps",
        "//tf_quant_finance/black_scholes/approximations",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "vanilla_prices",
    srcs = ["vanilla_prices.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/types",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "vanilla_prices_test",
    srcs = ["vanilla_prices_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
        # xla_cpu_jit xla dep,
    ],
)

py_library(
    name = "brownian_bridge",
    srcs = ["brownian_bridge.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "brownian_bridge_test",
    srcs = ["brownian_bridge_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "crr_binomial_tree",
    srcs = ["crr_binomial_tree.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "crr_binomial_tree_test",
    srcs = ["crr_binomial_tree_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "implied_vol_utils",
    srcs = ["implied_vol_utils.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "implied_vol_lib",
    srcs = ["implied_vol_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":implied_vol_approximation",
        ":implied_vol_newton_root",
        ":implied_vol_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "implied_vol_lib_test",
    size = "medium",
    srcs = ["implied_vol_lib_test.py"],
    python_version = "PY3",
    deps = [
        ":implied_vol_newton_root",
        ":implied_vol_utils",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "implied_vol_approximation",
    srcs = ["implied_vol_approximation.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "implied_vol_approximation_test",
    srcs = ["implied_vol_approximation_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "implied_vol_newton_root",
    srcs = ["implied_vol_newton_root.py"],
    srcs_version = "PY3",
    deps = [
        ":implied_vol_approximation",
        ":implied_vol_utils",
        "//tf_quant_finance/math/root_search:newton",
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "implied_vol_newton_root_test",
    size = "medium",
    srcs = ["implied_vol_newton_root_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "variance_swaps",
    srcs = ["variance_swaps.py"],
    srcs_version = "PY3",
    deps = [
        ":vanilla_prices",
        "//tf_quant_finance/math:diff_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "variance_swaps_test",
    srcs = ["variance_swaps_test.py"],
    python_version = "PY3",
    deps = [
        ":variance_swaps",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
        # xla_cpu_jit xla dep,
    ],
)

py_library(
    name = "asian_prices",
    srcs = ["asian_prices.py"],
    srcs_version = "PY3",
    deps = [
        ":vanilla_prices",
        "//tf_quant_finance/types",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "asian_prices_test",
    srcs = ["asian_prices_test.py"],
    python_version = "PY3",
    deps = [
        ":asian_prices",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
