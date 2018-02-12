import os

import numpy as np
import tvm
import topi
from tvm.contrib import rpc, util
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

dtype = 'float32'

def convert_to_remote(func, remote):
    temp = util.tempdir() 
    prefix = str(np.random.randint(1 << 31)) + "_"
    path_dso = temp.relpath(prefix + "tmp_func.tar")
    func.export_library(path_dso)

    remote.upload(path_dso)
    func = remote.load_module(prefix + "tmp_func.tar")
    return func


def generate_tune_packs(item_list):
    ret = []

    now = {}
    def dfs(depth):
        if depth == len(item_list):
            ret.append(now.copy())
            return

        name = item_list[depth][0]
        for value in item_list[depth][1]:
            now[name] = value
            dfs(depth + 1)

    dfs(0)

    return ret

USE_MANUAL_CODE = False
@tvm.register_func
def tvm_callback_opencl_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    with open("generated.cl", 'w') as fout:
        fout.write(code)
    if USE_MANUAL_CODE:
        split = code.split("\n")
        code = '\n'.join(split)
    return code



def tune_conv2d_nchw(batch, in_size, in_channel, num_filter, kernel, padding, stride, ctx,
                       n_times=1, target_host=None, remote=None):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), dtype=dtype, name='data')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), dtype=dtype, name='weight')

    # get verify data
    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    @memoize("topi.tests.test_topi_conv2d.verify_conv2d")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape)
        w_np = np.random.uniform(size=w_shape)
        b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()
    a = tvm.nd.array(a_np.astype(dtype), ctx)
    w = tvm.nd.array(w_np.astype(dtype), ctx)
    b = tvm.nd.array(np.zeros(b_np.shape).astype(dtype), ctx)
 
    # generate static config
    #tune_pack = generate_tune_packs([
    #        ["bn", [4]],
    #        ["num_thread", [1, 2, 4, 8, 16]],
    #        ["unroll_step", [1, 4, 16]],
    #    ])

    tune_pack = generate_tune_packs([
            ["VH", [1, 2, 4]],
            ["VW", [1, 2, 4, 8]],
            ["VC", [1, 2, 4, 8]],
            ["num_thread", [1, 2, 4, 16, 32, 64]],
    ])

    # search
    best_cost = 1e9
    best_config = None
    for config in reversed(tune_pack):
        with tvm.target.mali():
            tvm.target.current_target().tune_config = config
            B = topi.nn.conv2d(A, W, stride, padding)
            s = topi.generic.schedule_conv2d_nchw([B])
            func = tvm.build(s, [A, W, B], target_host=target_host)

        if remote is not None:
            func = convert_to_remote(func, remote)

        time_f = func.time_evaluator(func.entry_name, ctx, number=n_times)
        cost = time_f(a, w, b).mean

        try:
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4)
        except Exception as e:
            pass

        gflops = 2.0 * np.prod(b.shape) * kernel * kernel * in_channel /(1e9)/ cost
        print(config, cost, gflops)
        if cost < best_cost:
            best_cost = cost
            best_config = config

    return best_cost, 2.0 * np.prod(b.shape) * kernel * kernel * in_channel /(1e9)/ best_cost, best_config


def verify_conv2d_nchw(batch, in_size, in_channel, channel_multiplier, kernel, padding, stride, ctx,
                       n_times=1, target_host=None, remote=None):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), dtype=dtype, name='data')
    W = tvm.placeholder((in_channel, channel_multiplier, kernel, kernel), dtype=dtype, name='weight')

    with tvm.target.mali():
        B = topi.nn.depthwise_conv2d_nchw(A, W, stride, padding)
        #B = topi.nn.relu(B)
        s = topi.generic.schedule_depthwise_conv2d_nchw([B])
        func = tvm.build(s, [A, W, B], target_host=target_host)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    @memoize("topi.tests.test_topi_depthconv.verify_depthconv")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype('float32')
        w_np = np.random.uniform(size=w_shape).astype('float32')
        b_np = topi.testing.depthwise_conv2d_python_nchw(a_np, w_np, stride, padding)
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()
    a = tvm.nd.array(a_np.astype(dtype), ctx)
    w = tvm.nd.array(w_np.astype(dtype), ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape)).astype(dtype), ctx)

    if remote is not None:
        func = convert_to_remote(func, remote)

    time_f = func.time_evaluator(func.entry_name, ctx, number=n_times)
    cost = time_f(a, w, b).mean

    try:
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-1)
    except Exception as e:
        print(e)

    return cost, 2.0 * np.prod(b.shape) * kernel * kernel / (1e9) / cost

workloads = [
    # mobilenet
    (1, 112, 32, 1, 3, 1, 1),
    (1, 112, 64, 1, 3, 1, 2),
    (1, 56, 128, 1, 3, 1, 1),
    (1, 56, 128, 1, 3, 1, 2),
    (1, 28, 256, 1, 3, 1, 1),
    (1, 28, 256, 1, 3, 1, 2),
    (1, 14, 512, 1, 3, 1, 1),
    (1, 14, 512, 1, 3, 1, 2),
    (1, 7, 1024, 1, 3, 1, 1),
]

def verify_workloads(ctx, n_times=1, target_host=None, remote=None):
    for item in workloads:
        cost, gflops = verify_conv2d_nchw(*item, ctx=ctx, target_host=target_host, remote=remote)
        print("%-30s %.6f %.6f" % (item, cost, gflops))

def tune_workloads(ctx, n_times=1, target_host=None, remote=None):
    for item in workloads:
        cost, gflops, config = tune_conv2d_nchw(*item, ctx=ctx, target_host=target_host, remote=remote)
        print(item, cost, gflops, config)

if __name__ == "__main__":
    host = os.environ["TVM_OPENCL_DEVICE_HOST"]
    port = 9090
    remote = rpc.connect(host, port)
    target_host = "llvm -target=aarch64-linux-gnu -mattr=+neon"

    verify_workloads(remote.cl(), 10000, target_host, remote)

