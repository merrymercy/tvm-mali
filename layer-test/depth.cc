#include <chrono>
#include <iostream>
#include <iomanip>
#include <cassert>

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "util.h"

using namespace arm_compute;

struct Workload {
    std::string in_dtype;
    std::string out_dtype;
    size_t n;
    size_t height;
    size_t in_filter;
    int channel_m;
    size_t hkernel;
    size_t hpad;
    size_t hstride;
};

// measure the cost and gflops of 2d convolution
std::pair<double, double> MeasureConv(const Workload &w, int times=100) {
    assert(w.in_dtype == w.out_dtype);
    Format format = DtypeToFormat(w.in_dtype);

    CLTensor input, weight, output;
    PadStrideInfo conv_info(w.hstride, w.hstride, w.hpad, w.hpad);

    // init OpenCL
    CLScheduler::get().default_init();

    // allocate tensors
    input.allocator()->init(TensorInfo(TensorShape(w.height, w.height, w.in_filter), format));
    weight.allocator()->init(TensorInfo(TensorShape(w.hkernel, w.hkernel, w.in_filter), format));
    size_t h_out = (w.height - w.hkernel + w.hpad * 2) / w.hstride + 1;
    output.allocator()->init(TensorInfo(TensorShape(h_out, h_out, w.in_filter), format));
    input.allocator()->allocate();
    weight.allocator()->allocate();
    output.allocator()->allocate();
    CLScheduler::get().sync();

    // configure gemm function
    CLDepthwiseConvolutionLayer conv2d;
    conv2d.configure(&input, &weight, nullptr, &output, conv_info);

    // run test
    conv2d.run();
    CLScheduler::get().sync();
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < times; i++) {
        conv2d.run();
    }
    CLScheduler::get().sync();

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // calcuate gflops
    std::chrono::duration<double> fp_ms = end - begin;
    double cost = fp_ms.count() / times;
    return std::make_pair(cost, 2.0 * h_out * h_out *
                          w.hkernel * w.hkernel * w.in_filter / 1e9 / cost);
}


int main(int argc, const char **argv)
{
    Workload to_test[] = {
        // mobilenet
        Workload{"float32", "float32", 1, 112, 32, 1, 3, 1, 1},
        Workload{"float32", "float32", 1, 112, 64, 1, 3, 1, 2},
        Workload{"float32", "float32", 1, 56, 128, 1, 3, 1, 1},
        Workload{"float32", "float32", 1, 56, 128, 1, 3, 1, 2},
        Workload{"float32", "float32", 1, 28, 256, 1, 3, 1, 1},
        Workload{"float32", "float32", 1, 28, 256, 1, 3, 1, 2},
        Workload{"float32", "float32", 1, 14, 512, 1, 3, 1, 1},
        Workload{"float32", "float32", 1, 14, 512, 1, 3, 1, 2},
        Workload{"float32", "float32", 1, 7, 1024, 1, 3, 1, 1},
    };

    for (size_t i = 0; i < sizeof(to_test) / sizeof(to_test[0]); i++) {
        Workload &w = to_test[i];
        double cost, gflops;
        std::tie(cost, gflops) = MeasureConv(w);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << w.height << "x" << w.height << 'x' << w.in_filter << "x" << w.in_filter
                  << " " << w.hkernel << "\t";
        std::cout << "cost: " << cost  << ", "
                  << "GFLOPS: " << gflops << std::endl;
    }

    return 0;
}

