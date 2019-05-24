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

import argparse
import os
import subprocess

DEFAULT_REGION = 'us-west-2'


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--account')
    parser.add_argument('--version')
    parser.add_argument('--repo')
    parser.add_argument('--region', default=DEFAULT_REGION)

    return parser.parse_args()


args = _parse_args()

build_dir = os.path.join('docker', args.version, 'final')

# Run docker-login so we can pull the cached image
login_cmd = subprocess.check_output(
    'aws ecr get-login --no-include-email --registry-id {}'.format(args.account).split())
print('Executing docker login command: '.format(login_cmd))
subprocess.check_call(login_cmd.split())

for arch in ['cpu', 'gpu', 'eia']:
    for py_version in ['2.7', '3.6']:
        tag_arch = 'cpu' if arch == 'eia' else arch
        tag = '{}-{}-py{}'.format(args.version, tag_arch, py_version[0])
        repo = '{}-eia'.format(args.repo) if arch == 'eia' else args.repo
        dest = '{}:{}'.format(repo, tag)

        prev_image_uri = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(args.account, args.region, dest)
        dockerfile = os.path.join(build_dir, 'Dockerfile.{}'.format(arch))

        build_cmd = [
            'docker', 'build',
            '-f', dockerfile,
            '--cache-from', prev_image_uri,
            '--build-arg', 'py_version={}'.format(py_version),
            '-t', dest,
            '.',
        ]
        print('Building docker image: {}'.format(' '.join(build_cmd)))
        subprocess.check_call(build_cmd)