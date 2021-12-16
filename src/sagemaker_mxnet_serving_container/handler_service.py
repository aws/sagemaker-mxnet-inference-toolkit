# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import importlib
import logging
import os

import mxnet as mx
from sagemaker_inference import environment
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer

from sagemaker_mxnet_serving_container.default_inference_handler import DefaultGluonBlockInferenceHandler, \
    DefaultMXNetInferenceHandler
from sagemaker_mxnet_serving_container.mxnet_module_transformer import MXNetModuleTransformer

PYTHON_PATH_ENV = "PYTHONPATH"
logging.basicConfig(level=logging.ERROR)


class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.

    Determines specific default inference handlers to use based on the type MXNet model being used.

    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.

    Based on: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md

    """
    def __init__(self):
        self._service = None

    @staticmethod
    def _user_module_transformer(model_dir=environment.model_dir):
        try:
            user_module = importlib.import_module(environment.Environment().module_name)
        except ModuleNotFoundError as e:
            logging.error("import_module exception: {}".format(e))
            raise ValueError('import_module exception: {}'.format(e))

        if hasattr(user_module, 'transform_fn'):
            return Transformer(default_inference_handler=DefaultMXNetInferenceHandler())

        model_fn = getattr(user_module, 'model_fn', DefaultMXNetInferenceHandler().default_model_fn)

        model = model_fn(model_dir)
        if isinstance(model, mx.module.BaseModule):
            return MXNetModuleTransformer()
        elif isinstance(model, mx.gluon.block.Block):
            return Transformer(default_inference_handler=DefaultGluonBlockInferenceHandler())
        else:
            raise ValueError('Unsupported model type: {}. Did you forget to implement '
                             '`transform_fn` or `model_fn` in your entry-point?'
                             .format(model.__class__.__name__))

    def initialize(self, context):
        """Calls the Transformer method that validates the user module against
        the SageMaker inference contract.
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # add model_dir/code to python path
        code_dir_path = "{}:".format(model_dir + '/code')
        if PYTHON_PATH_ENV in os.environ:
            os.environ[PYTHON_PATH_ENV] = code_dir_path + os.environ[PYTHON_PATH_ENV]
        else:
            os.environ[PYTHON_PATH_ENV] = code_dir_path

        self._service = self._user_module_transformer(model_dir)
        super(HandlerService, self).initialize(context)
