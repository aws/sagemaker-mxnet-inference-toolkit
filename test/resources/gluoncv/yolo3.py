# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# flake8: noqa
import json
import os

import gluoncv as gcv
import mxnet as mx

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    """
    ctx = mx.cpu()
    net = gcv.model_zoo.get_model(
        'yolo3_darknet53_voc',
        pretrained=False,
        ctx=ctx)
    batchify = gcv.data.batchify.Stack()
    net.load_parameters(os.path.join(model_dir, 'yolo3_darknet53_voc.params'), mx.cpu(0))
    net.hybridize()
    def image_transform(im_bytes):
        """
        Apply image transformation to raw byte images
        """
        img = [mx.image.imdecode(bytes.fromhex(im)) for im in im_bytes]
        out = gcv.data.transforms.presets.yolo.transform_test(img)
        return out[0]

    return net, image_transform, batchify

def transform_fn(model, data, input_content_type, output_content_type):
    """
    Transform a request using the GluonNLP model. Called once per request.
    :param model: The Gluon model and the vocab
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    net, image_transform, batchify = model
    batch = json.loads(data)
    model_input = batchify(image_transform(batch))

    x = net(model_input)
    return (x[0].asnumpy().tolist(), x[1].asnumpy().tolist(), x[2].asnumpy().tolist())
