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

import os
import numpy

from sagemaker import utils
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.mxnet.model import MXNetModel
from sagemaker.predictor import RealTimePredictor, StringDeserializer, npy_serializer
from sagemaker.utils import sagemaker_timestamp

from test.integration import RESOURCE_PATH
from test.integration.sagemaker import timeout

DEFAULT_HANDLER_PATH = os.path.join(RESOURCE_PATH, "default_handlers")
MODEL_PATH = os.path.join(DEFAULT_HANDLER_PATH, "model.tar.gz")
SCRIPT_PATH = os.path.join(DEFAULT_HANDLER_PATH, "model", "code", "empty_module.py")

string_deserializer = StringDeserializer()


def test_hosting(sagemaker_session, ecr_image, instance_type, framework_version):
    prefix = "mxnet-serving/default-handlers"
    model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=prefix)
    model = MXNetModel(
        model_data,
        "SageMakerRole",
        SCRIPT_PATH,
        image=ecr_image,
        framework_version=framework_version,
        sagemaker_session=sagemaker_session,
    )

    endpoint_name = utils.unique_name_from_base("test-mxnet-serving")
    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)

        output = predictor.predict([[1, 2]])
        assert [[4.9999918937683105]] == output


def test_mme_hosting(sagemaker_session, ecr_image, instance_type, framework_version):
    prefix = "mxnet-serving/default-handlers"
    model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=prefix)

    timestamp = sagemaker_timestamp()
    endpoint_name = "test-mxnet-multimodel-endpoint-{}".format(timestamp)
    model_name = "test-mxnet-multimodel-{}".format(timestamp)

    mxnet_model = MXNetModel(
        model_data,
        "SageMakerRole",
        SCRIPT_PATH,
        image=ecr_image,
        framework_version=framework_version,
        sagemaker_session=sagemaker_session,
    )

    multi_data_model = MultiDataModel(
        name=model_name,
        model_data_prefix=model_data,
        model=mxnet_model,
        sagemaker_session=sagemaker_session,
    )

    multi_data_model.add_model(mxnet_model.model_data)

    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        multi_data_model.deploy(1, instance_type, endpoint_name=endpoint_name)

        predictor = RealTimePredictor(
            endpoint=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=npy_serializer,
            deserializer=string_deserializer,
        )

        for model in multi_data_model.list_models():
            data = numpy.zeros(shape=(1, 1, 28, 28))
            result = predictor.predict(data, target_model=model)
            assert result == "Invoked model: {}".format(model)

