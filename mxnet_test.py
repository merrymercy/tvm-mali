import os
import time
import sys
import argparse

import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download

input_size = 224

def test_module(model, dtype):
    assert dtype == 'float32'

    if model == 'vgg16':
        model_block = mx.gluon.model_zoo.vision.get_vgg(16, pretrained=False)
    elif model == 'mobilenet':
        model_block = mx.gluon.model_zoo.vision.get_mobilenet(1.0, pretrained=False)
    elif model == 'resnet18':
        model_block = mx.gluon.model_zoo.vision.get_resnet(version=1, num_layers=18, pretrained=False)
    else:
        raise RuntimeError("invalid model model " + model)
    model_block.collect_params().initialize(mx.init.Xavier())

    # define input and test function
    x = mx.nd.array(np.zeros((1, 3, input_size, input_size)))
    def measure(n_time):
        out = model_block(x).asnumpy()
        tic = time.time()
        for i in range(n_time):
            out = model_block(x).asnumpy()
        cost = time.time() - tic
        return cost / n_time

    # benchmark
    # print("============================================================")
    # print("model: %s, dtype: %s" % (model, dtype))

    num_warmup = 15
    num_test   = 80
    if model == 'mobilenet': # mobilenet is fast, need more runs for stable measureament
        num_warmup *= 4
        num_test   *= 4

    # warm up
    # print("warm up...")
    measure(num_warmup)

    # print("test..")
    cost = measure(num_test)
    # print("cost per image: %.4fs" % cost)

    print("backend: MXNet+OpenBLAS\tmodel: %s\tdtype: %s\tcost:%.4f" % (model, dtype, cost))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['vgg16', 'mobilenet', 'resnet18', 'all'])
    args = parser.parse_args()

    if args.model == 'all':
        for model in ['resnet18', 'mobilenet', 'vgg16']:
            test_module(model, 'float32')
            time.sleep(20)
    else:
        test_module(args.model, 'float32')

