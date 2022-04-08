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
"""Script to handle file processing requests and queuing work.

This binary does two main things:

1. It listens for job requests. These contain the gcs path to portfolio and
  market data files to be processed.
2. It downloads those file and queues it for downstream processing.

An example call using curl
curl --header "Content-Type: application/json" --request POST \
  --data '{"market-data-file": "gs://BUCKET_NAME/market_data_file_name", \
    "portfolio-file": "gs://BUCKET_NAME/portfolio_file_name", \
      "force-refresh": true }' http://localhost:8080/jobreq
"""  # pylint: disable=g-docstring-has-escape

import os
from os import path
import time
from typing import Dict, List, Tuple, Union

from common import datatypes
import dataclasses
import flask
import zmq

from google.cloud import storage

app = flask.Flask(__name__)

DOWNLOAD_BASE_PATH = '/var/tmp/downloads'

# Communicates with the container doing the calculations at IPC_PATH + IPC_NAME.
# To use shared memory, IPC_PATH can be changed to /dev/shm (and enabling it in
# docker run).
IPC_PATH = '/var/tmp/ipc'
IPC_NAME = 'jobs'

# Port at which it receives requests.
PORT = os.environ.get('PORT') or 8080

app.logger.setLevel('INFO')
app.logger.info(f'Downloaded files will be saved at {DOWNLOAD_BASE_PATH}')

if not path.exists(DOWNLOAD_BASE_PATH):
  os.makedirs(DOWNLOAD_BASE_PATH)

if not path.exists(IPC_PATH):
  os.makedirs(IPC_PATH)


# Initialize the IPC socket.
def init_socket():
  context = zmq.Context()
  sender = context.socket(zmq.PUSH)
  channel = 'ipc://' + path.join(IPC_PATH, IPC_NAME)
  app.logger.info(f'Pricer requests will be sent at {channel}')
  sender.bind(channel)
  return sender


SENDER = init_socket()


@dataclasses.dataclass
class JobRequest:
  """Specifies the request expected by the process route.

  Note that all underscores in field names are replaced by dashes in the JSON.
  """
  # Full GCS path (with gs:// prefix) to portfolio file.
  portfolio_file: str
  # Full GCS path (with gs:// prefix) to market data.
  market_data_file: str
  # Whether to force download of the files even if they exist locally.
  force_refresh: bool = False

  @classmethod
  def from_dict(cls, data: Dict[str, Union[str, bool]]) -> 'JobRequest':
    del cls
    return JobRequest(
        portfolio_file=data.get('portfolio-file', ''),
        market_data_file=data.get('market-data-file', ''),
        force_refresh=data.get('force-refresh', False))

  def to_dict(self) -> Dict[str, Union[str, bool]]:
    return {
        'portfolio-file': self.portfolio_file,
        'market-data-file': self.market_data_file,
        'force-refresh': self.force_refresh
    }

  def validate(self) -> Tuple[bool, str]:
    """Validates the data in the request."""
    is_ok = True
    messages: List[str] = []
    if not self.portfolio_file:
      is_ok = False
      messages.append('Portfolio file must be specified.')
    if not self.market_data_file:
      is_ok = False
      messages.append('Market data file must be specified.')
    return is_ok, ' '.join(messages)

  def market_data_local_path(self) -> str:
    return self._local_path_from_gcs_path(self.market_data_file)

  def portfolio_local_path(self) -> str:
    return self._local_path_from_gcs_path(self.portfolio_file)

  def market_data_exists_locally(self) -> bool:
    return path.exists(self.market_data_local_path())

  def portfolio_exists_locally(self) -> bool:
    return path.exists(self.portfolio_local_path())

  def download_if_needed(self) -> str:
    """Downloads specified data and returns an error message if any."""
    is_valid, err_message = self.validate()
    if not is_valid:
      return err_message
    if (not self.force_refresh and self.market_data_exists_locally() and
        self.portfolio_exists_locally()):
      return ''
    market_data_path = self.market_data_local_path()
    self._ensure_base_path_exists(market_data_path)
    self._download_from_gcs(self.market_data_file, market_data_path)
    portfolio_data_path = self.portfolio_local_path()
    self._ensure_base_path_exists(portfolio_data_path)
    self._download_from_gcs(self.portfolio_file, portfolio_data_path)
    return ''

  def _split_gcs_path(self, gcs_path: str) -> Tuple[str, str]:
    gcs_path = gcs_path.replace('gs://', '')  # Remove the prefix.
    pieces = gcs_path.split('/')
    bucket_name = pieces[0]
    remainder = '' if len(pieces) == 1 else path.join(*pieces[1:])
    return bucket_name, remainder

  def _ensure_base_path_exists(self, local_file_path: str) -> None:
    dir_path = os.path.dirname(local_file_path)
    if not path.exists(dir_path):
      os.makedirs(dir_path)

  def _download_from_gcs(self, gcs_file: str, local_file: str):
    client = storage.Client()
    bucket_name, blob_name = self._split_gcs_path(gcs_file)
    bucket = client.bucket(bucket_name)
    target_blob = bucket.blob(blob_name)
    start_time = time.time()
    target_blob.download_to_filename(local_file)
    end_time = time.time()
    app.logger.info('Downloaded %s to %s in %f seconds.', gcs_file, local_file,
                    end_time - start_time)

  def _local_path_from_gcs_path(self, gcs_path: str) -> str:
    return path.join(DOWNLOAD_BASE_PATH, gcs_path.replace('gs://', ''))


def _make_response(status, message):
  return flask.jsonify(dict(status=status, message=message))


@app.route('/jobreq', methods=['POST'])
def forward_request():
  """Forwards incoming job requests to the pricer."""
  request_data = flask.request.get_json()
  job_request = JobRequest.from_dict(request_data)
  err = job_request.download_if_needed()
  if err:
    return _make_response('Error', err), 400
  else:
    # Send work to the compute process.
    app.logger.info('Forwarding compute request to backend.')
    data_obj = datatypes.ComputeData(
        market_data_path=job_request.market_data_local_path(),
        portfolio_path=job_request.portfolio_local_path())
    SENDER.send_pyobj(data_obj)
    app.logger.info('Sent request to backend.')
    return _make_response('OK', ''), 201


@app.route('/')
def is_alive():
  return 'It is alive\n', 200


if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0', port=PORT)
