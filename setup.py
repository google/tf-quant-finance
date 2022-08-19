# Copyright 2019 Google LLC
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
# ==============================================================================
"""Setup for pip package."""

import datetime
from os import path
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

if sys.version_info[0] < 3:
  # Need to load open from io to support encoding arg when using Python 2.
  from io import open  # pylint: disable=redefined-builtin, g-importing-member, g-import-not-at-top

# Read the contents of the README file and set that as the long package
# description.
cwd = path.abspath(path.dirname(__file__))
with open(path.join(cwd, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

description = 'High-performance TensorFlow library for quantitative finance.'

major_version = '0'
minor_version = '0'
patch_version = '1'

if '--nightly' in sys.argv:
  # Run `python3 setup.py --nightly ...` to create a nightly build.
  sys.argv.remove('--nightly')
  project_name = 'tff-nightly'
  release_suffix = datetime.datetime.utcnow().strftime('.dev%Y%m%d')
  tfp_package = 'tensorflow-probability >= 0.12.1'
else:
  project_name = 'tf-quant-finance'
  # The suffix should be replaced with 'aN', 'bN', or 'rcN' (note: no dots) for
  # respective alpha releases, beta releases, and release candidates. And it
  # should be cleared, i.e. set to '', for stable releases (c.f. PEP 440).
  release_suffix = '.dev34'
  tfp_package = 'tensorflow-probability >= 0.12.1'

__version__ = '.'.join([major_version, minor_version, patch_version])
if release_suffix:
  __version__ += release_suffix

REQUIRED_PACKAGES = [
    'attrs >= 18.2.0', tfp_package, 'numpy >= 1.21', 'protobuf'
]


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False


setup(
    name=project_name,
    version=__version__,
    description=description,
    author='Google Inc.',
    author_email='tf-quant-finance@google.com',
    url='https://github.com/google/tf-quant-finance',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        'Operating System :: OS Independent',
    ],
    license='Apache 2.0',
    keywords='tensorflow quantitative finance hpc gpu option pricing',
    package_data={
        'tf_quant_finance': [
            'third_party/sobol_data/new-joe-kuo.6.21201',
            'third_party/sobol_data/LICENSE'
        ]
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
