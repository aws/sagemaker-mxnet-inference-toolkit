# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from glob import glob
import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker_mxnet_serving_container',
    version='1.0.0',
    description='Open source library for creating MXNet containers for serving on SageMaker.',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob('src/*.py')],

    long_description=read('README.rst'),
    author='Amazon Web Services',
    url='https://github.com/aws/sagemaker-mxnet-serving-container',
    license='Apache License 2.0',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],

    # We don't declare our dependency on mxnet here because we build with
    # different packages for different variants (e.g. mxnet-mkl and mxnet-cu90).
    install_requires=['sagemaker-inference==1.0.0'],
    extras_require={
        'test': ['tox', 'flake8', 'pytest', 'pytest-cov', 'pytest-xdist', 'mock',
                 'sagemaker==1.23.0', 'docker-compose', 'awslogs']
    },

    entry_points={
        'console_scripts': 'serve=sagemaker_mxnet_serving_container.serving:main'
    }
)
