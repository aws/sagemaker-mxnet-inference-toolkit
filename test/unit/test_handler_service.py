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

import mxnet as mx
import pytest
from mock import patch, Mock
from sagemaker_inference.transformer import Transformer

from sagemaker_mxnet_serving_container.handler_service import HandlerService
from sagemaker_mxnet_serving_container.mxnet_module_transformer import MXNetModuleTransformer


@patch('sagemaker_mxnet_serving_container.handler_service.HandlerService._user_module_transformer')
def test_handler_service(user_module_transformer):
    service = HandlerService()

    assert service._service == user_module_transformer()


class UserModuleTransformFn:
    def __init__(self):
        self.transform_fn = Mock()


@patch('importlib.import_module', return_value=UserModuleTransformFn())
def test_user_module_transform_fn(import_module):
    transformer = HandlerService._user_module_transformer()

    assert transformer._transform_fn == import_module.return_value.transform_fn
    assert isinstance(transformer, Transformer)


class UserModuleModelFn:
    def __init__(self):
        self.model_fn = Mock()


@patch('sagemaker_mxnet_serving_container.default_inference_handler.DefaultModuleInferenceHandler.default_predict_fn')
@patch('sagemaker_mxnet_serving_container.default_inference_handler.DefaultModuleInferenceHandler.default_input_fn')
@patch('importlib.import_module', return_value=UserModuleModelFn())
def test_user_module_mxnet_module_transformer(import_module, input_fn, predict_fn):
    import_module.return_value.model_fn.return_value = mx.module.BaseModule()

    transformer = HandlerService._user_module_transformer()

    assert isinstance(transformer, MXNetModuleTransformer)
    assert transformer._input_fn == input_fn
    assert transformer._predict_fn == predict_fn


@patch('sagemaker_mxnet_serving_container.default_inference_handler.DefaultMXNetInferenceHandler.default_model_fn')
@patch('sagemaker_mxnet_serving_container.default_inference_handler.DefaultGluonBlockInferenceHandler.default_predict_fn')
@patch('importlib.import_module', return_value=object())
def test_user_module_mxnet_gluon_transformer(import_module, predict_fn, model_fn):
    model_fn.return_value = mx.gluon.block.Block()

    transformer = HandlerService._user_module_transformer()

    assert isinstance(transformer, Transformer)
    assert transformer._predict_fn == predict_fn
    assert transformer._model_fn == model_fn


@patch('importlib.import_module', return_value=UserModuleModelFn())
def test_user_module_unsupported(import_module):
    with pytest.raises(ValueError) as e:
        HandlerService._user_module_transformer()

    assert 'Unsupported model type' in str(e)
