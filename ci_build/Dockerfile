FROM ubuntu:focal

# sudo isn't really needed, but we include it for convenience
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl wget build-essential rsync sudo python3 python3-dev python3-distutils python-is-python3 git

# Install the latest version of pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

# Install Bazel
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
RUN sudo apt-get update && sudo apt-get install -y bazel

# Install pip packages
RUN python3 -m pip install --upgrade numpy==1.22
RUN python3 -m pip install --upgrade keras==2.12.0rc1 tensorboard==2.12 tensorflow==2.12.0rc1 tensorflow-probability==0.19.0 attrs
RUN python3 -m pip install --upgrade protobuf==3.20.3

# Change work directory
WORKDIR /workspace

ENTRYPOINT ["bazel"]


