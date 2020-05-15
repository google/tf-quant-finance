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

# Builds an image for a container that receives job requests and downloads
# portfolio and market data files from GCS.

# To build the image, it is necessary that docker build be invoked from the
# parent folder of this file (i.e. from inside option_pricing_basic.).
# Example container execution command:
# $ docker build -f ./downloader/Dockerfile -t downloader .
# $ docker run --name download-app \
#   -v <PathToTheKeyFileFolder>:/tmp/google_app_creds \
#   -v /var/tmp/:/var/tmp/ -p 8080:8080 -it downloader

FROM python:3

COPY ./downloader/requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY ./downloader/app /app/
COPY ./common /app/common/

VOLUME ["/var/tmp/"]

# Uncomment this and set it so the gcs client API can use the credentials.
# User has to ensure that the keyfile is called keyfile.json and the path
# /tmp/google_app_creds in the container is mapped to the path to the
# file in the host. See sample docker run command above.
# If using the container in kubernetes, this should be set in the kubernetes
# settings if needed.
# ENV GOOGLE_APPLICATION_CREDENTIALS="/tmp/google_app_creds/keyfile.json"

# Listen on port 8080 for incoming job requests.
EXPOSE 8080/tcp


WORKDIR /app
ENTRYPOINT ["python3", "main.py"]
