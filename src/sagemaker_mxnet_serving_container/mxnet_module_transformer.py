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

import inspect

from sagemaker_inference.transformer import Transformer
from sagemaker_mxnet_serving_container.default_inference_handler import DefaultModuleInferenceHandler


class MXNetModuleTransformer(Transformer):
    """Custom ``Transformer``, which passes the model object to the input_fn,
    as required in the default_input_fn for Module based MXNet models.
    """
    def __init__(self):
        super(MXNetModuleTransformer, self).__init__(DefaultModuleInferenceHandler())

    def _default_transform_fn(self, model, input_data, content_type, accept):
        data = self._call_input_fn(input_data, content_type, model)
        prediction = self._predict_fn(data, model)
        result = self._output_fn(prediction, accept)
        return result

    # The default_input_fn for Modules requires access to the model object.
    # The input_fn contract only allows access to two parameters.
    # To not break input_fn that are only expecting two parameters.
    def _call_input_fn(self, input_data, content_type, model):
        try:  # PY3
            argspec = inspect.getfullargspec(self._input_fn)

        except AttributeError:  # PY2
            argspec = inspect.getargspec(self._input_fn)

        if 'model' in argspec.args:
            return self._input_fn(input_data, content_type, model)

        return self._input_fn(input_data, content_type)
