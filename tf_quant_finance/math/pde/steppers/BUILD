# Steppers for PDE solvers

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "steppers",
    srcs = [
        "__init__.py",
        "composite_stepper.py",
        "crank_nicolson.py",
        "douglas_adi.py",
        "explicit.py",
        "extrapolation.py",
        "implicit.py",
        "multidim_parabolic_equation_stepper.py",
        "oscillation_damped_crank_nicolson.py",
        "parabolic_equation_stepper.py",
        "weighted_implicit_explicit.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "parabolic_equation_stepper_test",
    size = "medium",
    timeout = "long",
    srcs = ["parabolic_equation_stepper_test.py"],
    python_version = "PY3",
    shard_count = 12,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "time_marching_schemes_test",
    size = "medium",
    srcs = ["time_marching_schemes_test.py"],
    python_version = "PY3",
    shard_count = 1,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "multidim_parabolic_equation_stepper_test",
    size = "medium",
    timeout = "long",
    srcs = ["multidim_parabolic_equation_stepper_test.py"],
    python_version = "PY3",
    shard_count = 12,
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_test(
    name = "douglas_adi_scheme_test",
    size = "medium",
    srcs = ["douglas_adi_scheme_test.py"],
    python_version = "PY3",
    shard_count = 1,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
