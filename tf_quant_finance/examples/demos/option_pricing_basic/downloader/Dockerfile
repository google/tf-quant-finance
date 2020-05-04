# Builds an image for the container that receives job requests and downloads
# portfolio and market data files from GCS.

# Example container execution command:
# docker build -t downloader .
# docker run --name <ContainerName> -p <HostPort>:8080 \
# -v <LocalServiceAccountKeyFilePath>:/tmp/google_app_creds -it downloader

FROM python:3

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY ./app /app

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
