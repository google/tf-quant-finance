# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tf_quant_finance",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tf_quant_finance/black_scholes",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental",
        "//tf_quant_finance/math",
        "//tf_quant_finance/models",
        "//tf_quant_finance/rates",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
    ],
)

# The "docs" filegroups define which files should be copied to the generated
# docs folder. For example, its convenient to have a README.md from each package
# there.
filegroup(
    name = "docs",
    data = [
        "//tf_quant_finance/black_scholes:docs",
        "//tf_quant_finance/datetime:docs",
        "//tf_quant_finance/experimental:docs",
        "//tf_quant_finance/math:docs",
        "//tf_quant_finance/models:docs",
        "//tf_quant_finance/rates:docs",
    ],
    visibility = ["__subpackages__"],
)

filegroup(
    name = "generated_docs",
    srcs = glob([
        "g3doc/**",
    ]),
    visibility = ["__subpackages__"],
)
