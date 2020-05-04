# Lint as: python3
"""Script to handle file processing requests and queuing work.

This binary does two main things:

1. It listens for job requests. These contain the gcs path to portfolio and
  market data files to be processed.
2. It downloads those file and queues it for downstream processing.

# TODO: implement queueing.

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

from absl import logging
import dataclasses
import flask

from google.cloud import storage

app = flask.Flask(__name__)

DOWNLOAD_FILES_LOC = (os.environ.get('APP_DOWNLOAD_FILES_LOC') or
                      '/var/tmp/app_downloads')

PORT = os.environ.get('PORT') or 8080

if not path.exists(DOWNLOAD_FILES_LOC):
  os.makedirs(DOWNLOAD_FILES_LOC)

logging.set_stderrthreshold('info')


@dataclasses.dataclass
class JobRequest:
  """Specifies the request expected by the process route.

  Note that all underscores in field names are replaced by dashes in the JSON.
  """
  portfolio_file: str  # Full GCS path (with gs:// prefix) to portfolio file.
  market_data_file: str  # Full GCS path (with gs:// prefix) to market data.
  force_refresh: bool = False  # Whether to force download of the files even if
                               # they exist locally.

  @classmethod
  def from_dict(cls, data: Dict[str, Union[str, bool]]) -> 'JobRequest':
    del cls
    return JobRequest(portfolio_file=data.get('portfolio-file', ''),
                      market_data_file=data.get('market-data-file', ''),
                      force_refresh=data.get('force-refresh', False))

  def to_dict(self) -> Dict[str, Union[str, bool]]:
    return {'portfolio-file': self.portfolio_file,
            'market-data-file': self.market_data_file,
            'force-refresh': self.force_refresh}

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
    return self._local_path_from_gcs_path(self.market_data_file)

  def market_data_exists_locally(self) -> bool:
    return path.exists(self.market_data_local_path())

  def portfolio_exists_locally(self) -> bool:
    return path.exists(self.portfolio_local_path())

  def download_if_needed(self) -> str:
    """Downloads specified data and returns an error message if any."""
    is_valid, err_message = self.validate()
    if not is_valid:
      return err_message
    if (not self.force_refresh and
        self.market_data_exists_locally() and
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
    logging.info('Downloaded %s to %s in %f seconds.',
                 gcs_file, local_file, end_time - start_time)

  def _local_path_from_gcs_path(self, gcs_path: str) -> str:
    return path.join(DOWNLOAD_FILES_LOC,
                     gcs_path.replace('gs://', ''))


def _make_response(status, message):
  return flask.jsonify(dict(status=status, message=message))


@app.route('/jobreq', methods=['POST'])
def process_request():
  request_data = flask.request.get_json()
  request = JobRequest.from_dict(request_data)
  err = request.download_if_needed()
  if err:
    return _make_response('Error', err), 400
  else:
    return _make_response('OK', ''), 201


@app.route('/')
def is_alive():
  return 'It is alive\n', 200


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=PORT)
