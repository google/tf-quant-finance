# Copyright 2020 Google LLC
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

# An image to price options using TFF.
# To build the image, it is necessary that docker build be invoked from the
# parent folder of this file (i.e. from inside option_pricing_basic.).
# Example container execution command:
# $ docker build -f ./pricer/Dockerfile -t pricer .
# $ docker run --name pricer-app -v /var/tmp/:/var/tmp/ -it pricer
# With default settings, the process running in the container will expect
# compute requests to arrive at ipc:///var/tmp/ipc/jobs.

FROM python:3

COPY ./pricer/requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY ./pricer/app /app/
COPY ./common /app/common/

VOLUME ["/var/tmp/"]

WORKDIR /app
ENTRYPOINT ["python3", "main.py"]
CMD ["--alsologtostderr"]
