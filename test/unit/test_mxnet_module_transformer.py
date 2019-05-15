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

from mock import Mock, patch

from sagemaker_mxnet_serving_container.mxnet_module_transformer import MXNetModuleTransformer

CONTENT_TYPE = 'content'
ACCEPT = 'accept'
DATA = 'data'
MODEL = 'foo'


@patch('importlib.import_module', return_value=object())
def test_default_transform_fn(import_module):
    preprocessed_data = 'preprocessed_data'
    prediction_result = 'result'
    processed_result = 'processed_result'

    input_fn = Mock(return_value=preprocessed_data)
    predict_fn = Mock(return_value=prediction_result)
    output_fn = Mock(return_value=processed_result)

    module_transformer = MXNetModuleTransformer()
    module_transformer._input_fn = input_fn
    module_transformer._predict_fn = predict_fn
    module_transformer._output_fn = output_fn

    result = module_transformer._default_transform_fn(MODEL, DATA, CONTENT_TYPE, ACCEPT)

    input_fn.assert_called_once_with(DATA, CONTENT_TYPE, MODEL)
    predict_fn.assert_called_once_with(preprocessed_data, MODEL)
    output_fn.assert_called_once_with(prediction_result, ACCEPT)
    assert processed_result == result
