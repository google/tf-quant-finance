# To build Docker image:
# docker build -t tf-quant-finance --no-cache .
#
# TODO(b/141456317): Allow caching and remove --no-cache parameter.
#
# To run Docker container:
# docker run -it tf-quant-finance

FROM ubuntu:latest

# sudo isn't really needed, but we include it for convenience
RUN apt-get update && apt-get install -y curl wget build-essential rsync vim openjdk-11-jdk sudo python3 python3-distutils git

# Install the latest version of pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

# Install bazel
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
RUN sudo apt-get update && sudo apt-get install -y bazel

# Install pip packages
RUN pip install --upgrade tensorflow==2.2 tensorflow-probability==0.9 numpy==1.16 attrs

# Clone GitHub repository
RUN git clone https://github.com/google/tf-quant-finance.git tf-quant-finance

# Change workdir
WORKDIR /tf-quant-finance
