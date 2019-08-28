#!/bin/bash

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
# ==============================================================================
function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

# Check if we are building GPU or CPU ops, default CPU
while [[ "$TF_NEED_CUDA" == "" ]]; do
  read -p "Build against TensorFlow CPU package?"\
" Y or enter for CPU (tensorflow), N for GPU (tensorflow-gpu). [Y/n] " INPUT
  case $INPUT in
    [Yy]* ) echo "Build with CPU pip package."; TF_NEED_CUDA=0;;
    [Nn]* ) echo "Build with GPU pip package."; TF_NEED_CUDA=1;;
    "" ) echo "Build with CPU pip package."; TF_NEED_CUDA=0;;
    * ) echo "Invalid selection: " $INPUT;;
  esac
done

# Check if we are building against manylinux1 or manylinux2010 pip package,
# default manylinux1
while [[ "$PIP_MANYLINUX2010" == "" ]]; do
  read -p "Does the pip package have tag manylinux2010 (usually the case for nightly release after Aug 1, 2019, or official releases past 1.14.0)?"\
" Y or enter for manylinux2010, N for manylinux1. [y/N] " INPUT
  case $INPUT in
    [Yy]* ) echo "Build against pip package with manylinux2010 tag. --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain will be added to bazel command."; PIP_MANYLINUX2010=1;;
    [Nn]* ) echo "Build against pip package with manylinux1."; PIP_MANYLINUX2010=0;;
    "" ) echo "Build against pip package with manylinux1."; PIP_MANYLINUX2010=0;;
    * ) echo "Invalid selection: " $INPUT;;
  esac
done

# CPU
if [[ "$TF_NEED_CUDA" == "0" ]]; then

  # Check if it's installed
  if [[ $(pip show tensorflow) == *tensorflow* ]] || [[ $(pip show tf-nightly) == *tf-nightly* ]] ; then
    echo 'Using installed tensorflow'
  else
    # Uninstall GPU version if it is installed.
    if [[ $(pip show tensorflow-gpu) == *tensorflow-gpu* ]]; then
      echo 'Already have gpu version of tensorflow installed. Uninstalling......\n'
      pip uninstall tensorflow-gpu
    elif [[ $(pip show tf-nightly-gpu) == *tf-nightly-gpu* ]]; then
      echo 'Already have gpu version of tensorflow installed. Uninstalling......\n'
      pip uninstall tf-nightly-gpu
    fi
    # Install CPU version
    echo 'Installing tensorflow......\n'
    pip install tensorflow
  fi

else

  # Check if it's installed
   if [[ $(pip show tensorflow-gpu) == *tensorflow-gpu* ]] || [[ $(pip show tf-nightly-gpu) == *tf-nightly-gpu* ]]; then
    echo 'Using installed tensorflow-gpu'
  else
    # Uninstall CPU version if it is installed.
    if [[ $(pip show tensorflow) == *tensorflow* ]]; then
      echo 'Already have tensorflow non-gpu installed. Uninstalling......\n'
      pip uninstall tensorflow
    elif [[ $(pip show tf-nightly) == *tf-nightly* ]]; then
      echo 'Already have tensorflow non-gpu installed. Uninstalling......\n'
      pip uninstall tf-nightly
    fi
    # Install CPU version
    echo 'Installing tensorflow-gpu .....\n'
    pip install tensorflow-gpu
  fi
fi

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
if [[ "$PIP_MANYLINUX2010" == "0" ]]; then
  write_to_bazelrc "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"
fi

write_to_bazelrc "build:manylinux2010 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain"
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  else
    SHARED_LIBRARY_NAME="libtensorflow_framework.so"
  fi
fi
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}

if [[ "$TF_NEED_CUDA" == "1" ]]; then
  write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "/usr/lib/x86_64-linux-gnu"
  write_action_env_to_bazelrc "TF_CUDA_VERSION" "10.0"
  write_action_env_to_bazelrc "TF_CUDNN_VERSION" "7"
  write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "/usr/local/cuda"
  write_to_bazelrc "build --config=cuda"
  write_to_bazelrc "test --config=cuda"
fi

if [[ "$PIP_MANYLINUX2010" == "1" ]]; then
  write_to_bazelrc "build --config=manylinux2010"
  write_to_bazelrc "test --config=manylinux2010"
fi
