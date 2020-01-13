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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

__version__ = '0.0.1dev11'
REQUIRED_PACKAGES = [
    'attrs >= 18.2.0', 'tensorflow-probability >= 0.8.0', 'numpy >= 1.16.0'
]

project_name = 'tf-quant-finance'
description = 'High-performance TensorFlow library for quantitative finance.'


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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
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
