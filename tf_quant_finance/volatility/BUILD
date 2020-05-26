# Tools for building volitility Models

licenses(["notice"])
package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
	name="volitility",
	srcs= ["__init__.py"],
	deps = [
	     ":barrier_option"
	],
)

py_library(
	name="barrier_option",
	srcs=["barrier_option.py"],
	deps=[]
)

py_test(
	name="barrier_option_test",
	srcs=["barrier_option_test.py"],
	python_version="PY3",
	deps=[
		":volitility",
		"//tf_quant_finance",
	],
)