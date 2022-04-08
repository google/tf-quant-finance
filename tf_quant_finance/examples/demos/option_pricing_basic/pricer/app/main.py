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
"""Computes option prices using TensorFlow Finance."""

import os
from os import path

from absl import app
from absl import logging
from common import datatypes
import pricers
import tf_quant_finance as tff
import zmq

RESULTS_BASE_PATH = '/var/tmp/results'

if not path.exists(RESULTS_BASE_PATH):
  os.makedirs(RESULTS_BASE_PATH)

# Communicates with the container doing the calculations at IPC_PATH + IPC_NAME.
# To use shared memory, IPC_PATH can be changed to /dev/shm (and enabling it in
# docker run).
IPC_PATH = '/var/tmp/ipc'
IPC_NAME = 'jobs'

# The size of the option batches to be priced. If this is 0 or negative,
# then variable sized inputs can be supplied but it is more efficient to set
# an explicit batch size here.
BATCH_SIZE = os.environ.get('OPTION_BATCH_SIZE', 1000000)

# The number of assets on which the options are written. If this is 0 or
# negative, the market data can contain variable number of assets (across
# different job requests). It is more efficient to set an explicit value here.
NUM_ASSETS = os.environ.get('NUM_ASSETS', 1000)

logging.set_verbosity(logging.INFO)


def main(argv):
  del argv
  batch_size = None if BATCH_SIZE <= 0 else BATCH_SIZE
  num_assets = None if NUM_ASSETS <= 0 else NUM_ASSETS
  pricer = pricers.TffOptionPricer(batch_size=batch_size, num_assets=num_assets)
  context = zmq.Context()
  receiver = context.socket(zmq.PULL)
  channel = 'ipc://' + path.join(IPC_PATH, IPC_NAME)
  receiver.connect(channel)
  logging.info('Pricer ready and listening at %s', channel)

  while True:
    logging.info('Waiting for inputs...')
    inputs: datatypes.ComputeRequest = receiver.recv_pyobj()
    print('Received input data...')
    with pricers.Timer() as read_timer:
      market_data = tff.experimental.io.ArrayDictReader(
          inputs.market_data_path).next()
      portfolio = tff.experimental.io.ArrayDictReader(
          inputs.portfolio_path).next()
    logging.info('Read input data in %f ms', read_timer.elapsed_ms)
    with pricers.Timer() as compute_timer:
      prices = pricer.price(market_data['spot'], market_data['volatility'],
                            market_data['rate'], portfolio['underlier_id'],
                            portfolio['strike'], portfolio['call_put_flag'],
                            portfolio['expiry_date'])
      results = {'trade_id': portfolio['trade_id'], 'prices': prices}
      output_path = path.join(RESULTS_BASE_PATH,
                              'results_' + path.basename(inputs.portfolio_path))
    logging.info('Computed prices in %f ms', compute_timer.elapsed_ms)
    with pricers.Timer() as write_timer:
      with tff.experimental.io.ArrayDictWriter(output_path) as writer:
        writer.write(results)
    logging.info('Wrote output results to %s in %f ms', output_path,
                 write_timer.elapsed_ms)

    total_ms = (
        read_timer.elapsed_ms + compute_timer.elapsed_ms +
        write_timer.elapsed_ms)
    logging.info('Processed request of size %d in %f total ms.',
                 portfolio['underlier_id'].size, total_ms)


if __name__ == '__main__':
  app.run(main)
