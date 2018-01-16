import os
import time

import numpy as np
import tvm
import topi
from tvm.contrib import rpc, util
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

dtype = 'float16'

USE_MANUAL_CODE = False
@tvm.register_func
def tvm_callback_opencl_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    with open("generated.cl", 'w') as fout:
        fout.write(code)
    if USE_MANUAL_CODE:
        with open("manual.cl") as fin:
            code = "\n".join(fin.readlines())
        print(code)
    return code


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


def tune_dense(batch, hidden, out, ctx,
                       n_times=1, target_host=None, remote=None):
    A = tvm.placeholder((1, hidden), dtype=dtype, name='A')
    B = tvm.placeholder((out, hidden), dtype=dtype, name='B')
    BIAS = tvm.placeholder((out,), dtype=dtype, name='bias')

    # generate static config
    tune_pack = generate_tune_packs([
#            ["bn", [1, 2, 4, 8, 16]],
#            ["reuse", [1, 2, 4, 8]],
            ["num_thread", [1, 2, 4, 32, 64, 256]],
            ["unroll_step", [1, 2, 4, 5, 6, 16, 32]],
        ])

    a_shape = get_const_tuple(A.shape)
    b_shape = get_const_tuple(B.shape)
    bias_shape = get_const_tuple(BIAS.shape)
    c_shape = (1, out)

    a_np = np.random.uniform(size=a_shape)
    b_np = np.random.uniform(size=b_shape)
    bias_np = np.random.uniform(size=bias_shape)
    c_np = np.random.uniform(size=c_shape)

    # search
    tic = time.time()
    best_cost = 1e9
    best_config = None
    for i, config in enumerate(tune_pack):
        with tvm.target.mali():
            tvm.target.current_target().tune_config = config
            C = topi.nn.dense(A, B, BIAS)
            s = topi.generic.schedule_dense([C])
            func = tvm.build(s, [A, B, BIAS, C], target_host=target_host)

        a = tvm.nd.array(a_np.astype(dtype), ctx=ctx)
        b = tvm.nd.array(b_np.astype(dtype), ctx=ctx)
        bias = tvm.nd.array(bias_np.astype(dtype), ctx=ctx)
        c = tvm.nd.array(c_np.astype(dtype), ctx=ctx)

        if remote is not None:
            func = convert_to_remote(func, remote)

        time_f = func.time_evaluator(func.entry_name, ctx, number=n_times)
        cost = time_f(a, b, bias, c).mean

        gflops = 2.0 * np.prod(b.shape) / (1e9) / cost
        if cost < best_cost:
            print(config, cost, gflops)
            best_cost = cost
            best_config = config

        if i % 20 == 0:
            print(i, len(tune_pack), time.time()- tic, (time.time() - tic) / (i+1))

    try:
        np.testing.assert_allclose(np.dot(a_np, b_np.T) + bias_np, c.asnumpy(), rtol=1e-2)
    except Exception as e:
        pass
        print(e)

    return best_cost, 2.0 * np.prod(b.shape) / (1e9) / best_cost, best_config

def verify_dense(batch, hidden, out, ctx,
                       n_times=1, target_host=None, remote=None):
    A = tvm.placeholder((1, hidden),   dtype=dtype, name='A')
    B = tvm.placeholder((out, hidden), dtype=dtype, name='B')
    bias = tvm.placeholder((out,),     dtype=dtype, name='bias')

    with tvm.target.mali():
        C = topi.nn.dense(A, B, bias)
        s = topi.generic.schedule_dense([C])
        func = tvm.build(s, [A, B, bias, C], target_host=target_host)

    a_shape = get_const_tuple(A.shape)
    b_shape = get_const_tuple(B.shape)
    bias_shape = get_const_tuple(bias.shape)
    c_shape = get_const_tuple(C.shape)

    a_np = np.random.uniform(size=a_shape)
    b_np = np.random.uniform(size=b_shape)
    bias_np = np.random.uniform(size=bias_shape)
    c_np = np.random.uniform(size=c_shape)

    a = tvm.nd.array(a_np.astype(dtype), ctx=ctx)
    b = tvm.nd.array(b_np.astype(dtype), ctx=ctx)
    bias = tvm.nd.array(bias_np.astype(dtype), ctx=ctx)
    c = tvm.nd.array(np.zeros_like(c_np).astype(dtype), ctx=ctx)

    if remote is not None:
        func = convert_to_remote(func, remote)

    time_f = func.time_evaluator(func.entry_name, ctx, number=n_times)
    cost = time_f(a, b, bias, c).mean

    try:
        np.testing.assert_allclose(np.dot(a_np, b_np.T) + bias_np, c.asnumpy(), rtol=1e-1)
    except Exception as e:
        pass
        print(e)

    return cost, 2.0 * np.prod(b.shape) / (1e9) / cost

workloads = [
    (1, 25088, 4096),
#    (1, 4096, 4096),
#    (1, 4096, 1000),
#    (1, 1024, 1000),
]

def verify_workloads(ctx, n_times=1, target_host=None, remote=None):
    for item in workloads:
        cost, gflops = verify_dense(*item, ctx=ctx, target_host=target_host, remote=remote)
        print("%-30s %.6f %.6f" % (item, cost, gflops))

def tune_workloads(ctx, n_times=1, target_host=None, remote=None):
    for item in workloads:
        cost, gflops, config = tune_dense(*item, ctx=ctx, target_host=target_host, remote=remote)
        print(item, cost, gflops, config)

if __name__ == "__main__":
    host = os.environ["TVM_OPENCL_DEVICE_HOST"]
    port = 9090
    remote = rpc.connect(host, port)
    target_host = "llvm -target=aarch64-linux-gnu"

    verify_workloads(remote.cl(), 10, target_host, remote)

