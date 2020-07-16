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
# ============================================================================
workspace(name = "tf_quant_finance")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


# Load repository with py_proto_library definition
http_archive(
    name = "com_google_protobuf",
    sha256 = "bba884e1dce7b9ba5bce566bfa2d23ea2c824aa928a9133139c5485cefa69139",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/9ce8c330e736b99c2bb00bdc6a60f00ab600c283.zip"],
    strip_prefix = "protobuf-9ce8c330e736b99c2bb00bdc6a60f00ab600c283",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()



