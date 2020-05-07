# Builds an image for a container that receives job requests and downloads
# portfolio and market data files from GCS.

# To build the image, it is necessary that docker build be invoked from the
# parent folder of this file (i.e. from inside option_pricing_basic.).
# Example container execution command:
# $ docker build -f ./downloader/Dockerfile -t downloader .
# $ docker run --name download-app \
#   -v <PathToTheKeyFileFolder>:/tmp/google_app_creds \
#   -v /var/tmp:/var/tmp -p 8080:8080 -it downloader

FROM python:3

COPY ./downloader/requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY ./downloader/app /app/
COPY ./common /app/common/

VOLUME ["/var/tmp/downloads", "/var/tmp/ipc"]

# Set this so the gcs client API can use the credentials.
# User has to ensure that the keyfile is called keyfile.json and the path
# /tmp/google_app_creds in the container is mapped to the path to the
# file in the host. See sample docker run command above.
ENV GOOGLE_APPLICATION_CREDENTIALS="/tmp/google_app_creds/keyfile.json"

# Listen on port 8080.
EXPOSE 8080/tcp

WORKDIR /app
ENTRYPOINT ["python3", "main.py"]
CMD ["--logtostderr"]
