# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import json
import os
import tempfile

import pytest
from sagemaker import utils
from sagemaker.mxnet.model import MXNetModel

from test.integration import RESOURCE_PATH
from test.integration.sagemaker import timeout

GLUONCV_PATH = os.path.join(RESOURCE_PATH, 'gluoncv')
SCRIPT_PATH = os.path.join(GLUONCV_PATH, 'yolo3.py')
SCRIPT_DATA_PATH = os.path.join(GLUONCV_PATH, 'dog.jpg')


@pytest.mark.skip_py2_containers
def test_gluoncv(sagemaker_session, ecr_image, instance_type, framework_version):
    try:  # python3
        from urllib.request import urlretrieve
    except:  # python2
        from urllib import urlretrieve
    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, 'yolo3_darknet53_voc.tar.gz')
    urlretrieve('https://dlc-samples.s3.amazonaws.com/mxnet/gluon/yolo3_darknet53_voc.tar.gz', tmpfile)
    prefix = 'gluoncv-serving/default-handlers'
    model_data = sagemaker_session.upload_data(path=tmpfile, key_prefix=prefix)

    model = MXNetModel(model_data,
                       'SageMakerRole',
                       SCRIPT_PATH,
                       image=ecr_image,
                       py_version="py3",
                       framework_version=framework_version,
                       sagemaker_session=sagemaker_session)

    endpoint_name = utils.unique_name_from_base('test-mxnet-gluoncv')
    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)
        with open(SCRIPT_DATA_PATH, 'rb') as fdata:
            output = predictor.predict(json.dumps([fdata.read().hex()]))
        assert output[0][0].size == 100
