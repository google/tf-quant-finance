# Builds an image for a container that generates random options portfolio and
# associated market data in a binary format. It can also upload that data to
# a supplied bucket on GCS.

# To build the image, it is necessary that docker build be invoked from the
# parent folder of this file (i.e. from inside option_pricing_basic.).
# Example container execution command:
# $ docker build -f ./data_generator/Dockerfile -t datagen .
# $ docker run -v <PathToTheKeyFileFolder>:/tmp/google_app_creds \
# -v /tmp:/tmp -it datagen

FROM python:3

COPY ./data_generator/requirements.txt /app/
COPY ./data_generator/requirements_nodeps.txt /app/

# Splitting of requirements into two pieces is necessary to ensure that the
# tfp-nightly gets installed with TFF.
RUN pip install -r /app/requirements.txt
RUN pip install --no-deps -r /app/requirements_nodeps.txt

COPY ./data_generator/*.py /app/
COPY ./common /app/common/

# Set this so the gcs client API can use the credentials.
# User has to ensure that the keyfile is called keyfile.json and the path
# /tmp/google_app_creds in the container is mapped to the path to the
# file in the host. See sample docker run command above.
ENV GOOGLE_APPLICATION_CREDENTIALS="/tmp/google_app_creds/keyfile.json"

WORKDIR /app
ENTRYPOINT ["python3", "data_generation.py"]

# At a minimum, the user should override the output_path to either a GCS
# location or to a local path. In the latter case, a host volume mapping should
# be specified so the output files are reachable.
CMD ["--output_path=/tmp", "--num_underliers=1000", \
     "--options_per_file=1000000", "--num_files=50"]
