"""
Benchmark inference speed on ImageNet
Example (run on Firefly RK3399):
python mali_imagenet_bench.py --target-host 'llvm -target=aarch64-linux-gnu' --host 192.168.0.100 --port 9090 --model mobilenet
"""

import time
import argparse
import numpy as np
import tvm
import nnvm.compiler
import nnvm.testing
from tvm.contrib import util, rpc
from tvm.contrib import graph_runtime as runtime

def run_case(model, dtype):
    # load model
    if model == 'vgg16':
        net, params = nnvm.testing.vgg.get_workload(num_layers=16,
            batch_size=1, image_shape=image_shape, dtype=dtype)
    elif model == 'resnet18':
        net, params = nnvm.testing.resnet.get_workload(num_layers=18,
            batch_size=1, image_shape=image_shape, dtype=dtype)
    elif model == 'mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(
            batch_size=1, image_shape=image_shape, dtype=dtype)
    else:
        raise ValueError('no benchmark prepared for {}.'.format(model))

    # compile
    opt_level = 2 if dtype == 'float32' else 1
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, tvm.target.mali(), shape={"data": data_shape}, params=params,
            dtype=dtype, target_host=args.target_host)

    # upload model to remote device
    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    lib.export_library(lib_fname)
    remote = rpc.connect(args.host, args.port)
    remote.upload(lib_fname)

    ctx = remote.cl(0)
    rlib = remote.load_module('net.tar')
    rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

    # create graph runtime
    module = runtime.create(graph, rlib, ctx)
    module.set_input('data', tvm.nd.array(np.random.uniform(size=(data_shape)).astype(dtype)))
    module.set_input(**rparams)

    # benchmark
    print("============================================================")
    print("model: %s, dtype: %s" % (model, dtype))

    # the num of runs for warm up and test
    num_warmup = 10
    num_test   = 60
    if model == 'mobilenet': # mobilenet is fast, need more runs for stable measureament
        num_warmup *= 5
        num_test   *= 5

    # perform some warm up runs
    print("warm up..")
    warm_up_timer = module.module.time_evaluator("run", ctx, num_warmup)
    warm_up_timer()

    # test
    print("test..")
    ftimer = module.module.time_evaluator("run", ctx, num_test)
    prof_res = ftimer()
    print("cost per image: %.4fs" % prof_res.mean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['vgg16', 'resnet18', 'mobilenet', 'all'],
                        help="The model type.")
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'])
    parser.add_argument('--host', type=str, required=True, help="The host address of your arm device.")
    parser.add_argument('--port', type=int, required=True, help="The port number of your arm device")
    parser.add_argument('--target-host', type=str, required=True, help="The compilation target of host device.")
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    # set parameter
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)

    # load model
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)

    if args.model == 'all': # test all
        for model in ['vgg16', 'resnet18', 'mobilenet']:
            for dtype in ['float32', 'float16']:
                run_case(model, dtype)
                time.sleep(10)

    else:  # test single
        run_case(args.model, args.dtype)

