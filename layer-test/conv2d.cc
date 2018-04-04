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
    size_t batch;
    size_t height;
    size_t width;
    size_t in_filter;
    size_t out_filter;
    size_t hkernel;
    size_t wkernel;
    size_t hpad;
    size_t wpad;
    size_t hstride;
    size_t wstride;
};

// measure the cost and gflops of 2d convolution
std::pair<double, double> MeasureConv(const Workload &w, int times=30) {
    assert(w.in_dtype == w.out_dtype);
    Format format = DtypeToFormat(w.in_dtype);

    CLTensor input, weight, output;
    PadStrideInfo conv_info(w.wstride, w.hstride, w.wpad, w.hpad);

    // init OpenCL
    CLScheduler::get().default_init();

    // allocate tensors
    input.allocator()->init(TensorInfo(TensorShape(w.width, w.height, w.in_filter, w.batch), format));
    weight.allocator()->init(TensorInfo(TensorShape(w.wkernel, w.hkernel, w.in_filter, w.out_filter), format));
    size_t w_out = (w.width - w.wkernel + w.wpad * 2) / w.wstride + 1;
    size_t h_out = (w.height - w.hkernel + w.hpad * 2) / w.hstride + 1;
    output.allocator()->init(TensorInfo(TensorShape(w_out, h_out, w.out_filter, w.batch), format));
    input.allocator()->allocate();
    weight.allocator()->allocate();
    output.allocator()->allocate();
    CLScheduler::get().sync();

    // configure conv2d function
    CLConvolutionLayer conv2d;
    conv2d.configure(&input, &weight, nullptr, &output, conv_info);

    // run test
    conv2d.run();
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < times; i++) {
        conv2d.run();
    }
    CLScheduler::get().sync();

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // calcuate gflops
    std::chrono::duration<double> fp_ms = end - begin;
    double cost = fp_ms.count() / times;
    return std::make_pair(cost, 2.0 * w.batch * w_out * h_out * w.out_filter *
                          w.hkernel * w.wkernel * w.in_filter / 1e9 / cost);
}


int main(int argc, const char **argv)
{
    Workload to_test[] = {
        // vgg16
//        Workload{"float32", "float32", 1, 224, 224,  3, 64, 3, 3, 1, 1, 1, 1},
//        Workload{"float32", "float32", 1, 224, 224, 64, 64, 3, 3, 1, 1, 1, 1},
//        Workload{"float32", "float32", 1, 112, 112, 64, 128,3, 3, 1, 1, 1, 1},
//        Workload{"float32", "float32", 1, 112, 112,128, 128,3, 3, 1, 1, 1, 1},
//        Workload{"float32", "float32", 1, 56, 56, 128, 256, 3, 3, 1, 1, 1, 1},
//        Workload{"float32", "float32", 1, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1},
//        Workload{"float32", "float32", 1, 28, 28, 256, 512, 3, 3, 1, 1, 1, 1},
//        Workload{"float32", "float32", 1, 28, 28, 512, 512, 3, 3, 1, 1, 1, 1},
//        Workload{"float32", "float32", 1, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1},

        // resnet
        Workload{"float32", "float32",  1, 224, 224, 3, 64,  7, 7, 3, 3, 2, 2},
        Workload{"float32", "float32", 32, 224, 224, 3, 64,  7, 7, 3, 3, 2, 2},
        Workload{"float32", "float32",  1, 56, 56, 64,  64,  3, 3, 1, 1, 1, 1},
        Workload{"float32", "float32", 32, 56, 56, 64,  64,  3, 3, 1, 1, 1, 1},
        Workload{"float32", "float32",  1, 56, 56, 64,  64,  1, 1, 0, 0, 1, 1},
        Workload{"float32", "float32", 32, 56, 56, 64,  64,  1, 1, 0, 0, 1, 1},
        Workload{"float32", "float32",  1, 56, 56, 64,  128, 3, 3, 1, 1, 2, 2},
        Workload{"float32", "float32", 32, 56, 56, 64,  128, 3, 3, 1, 1, 2, 2},
        Workload{"float32", "float32",  1, 56, 56, 64,  128, 1, 1, 0, 0, 2, 2},
        Workload{"float32", "float32", 32, 56, 56, 64,  128, 1, 1, 0, 0, 2, 2},
        Workload{"float32", "float32",  1, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1},
        Workload{"float32", "float32", 32, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1},
        Workload{"float32", "float32",  1, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2},
        Workload{"float32", "float32", 32, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2},
        Workload{"float32", "float32",  1, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2},
        Workload{"float32", "float32", 32, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2},
        Workload{"float32", "float32",  1, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1},
        Workload{"float32", "float32", 32, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1},
        Workload{"float32", "float32",  1, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2},
        Workload{"float32", "float32", 32, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2},
        Workload{"float32", "float32",  1, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2},
        Workload{"float32", "float32", 32, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2},
        Workload{"float32", "float32",  1, 7,  7,  512, 512, 3, 3, 1, 1, 1, 1},
        Workload{"float32", "float32", 32, 7,  7,  512, 512, 3, 3, 1, 1, 1, 1},

//        // mobilenet
//        Workload{"float32", "float32", 1, 224, 224,   3,  32, 3, 3, 1, 1, 2, 2},
//        Workload{"float32", "float32", 1, 112, 112,  32,  64, 1, 1, 0, 0, 1, 1},
//        Workload{"float32", "float32", 1,  56,  56,  64, 128, 1, 1, 0, 0, 1, 1},
//        Workload{"float32", "float32", 1,  56,  56, 128, 128, 1, 1, 0, 0, 1, 1},
//        Workload{"float32", "float32", 1,  28,  28, 128, 256, 1, 1, 0, 0, 1, 1},
//        Workload{"float32", "float32", 1,  28,  28, 256, 256, 1, 1, 0, 0, 1, 1},
//        Workload{"float32", "float32", 1,  14,  14, 256, 512, 1, 1, 0, 0, 1, 1},
//        Workload{"float32", "float32", 1,  14,  14, 512, 512, 1, 1, 0, 0, 1, 1},
//        Workload{"float32", "float32", 1,   7,  7, 512, 1024, 1, 1, 0, 0, 1, 1},
//        Workload{"float32", "float32", 1,   7,  7, 1024,1024, 1, 1, 0, 0, 1, 1},
    };

    for (size_t i = 0; i < sizeof(to_test) / sizeof(to_test[0]); i++) {
        Workload &w = to_test[i];
        double cost, gflops;
        std::tie(cost, gflops) = MeasureConv(w);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << w.height << "x" << w.width << 'x' << w.in_filter << "x" << w.out_filter
                  << " " << w.hkernel << "\t";
        std::cout << "cost: " << cost  << ", "
                  << "GFLOPS: " << gflops << std::endl;
    }

    return 0;
}

