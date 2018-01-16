#include <chrono>
#include <iostream>
#include <iomanip>

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "util.h"

using namespace arm_compute;

// measure the cost and gflops of gemm
std::pair<double, double> MeasureGemm(int n, int l, int m, std::string dtype, int times=30) {
    Format format = DtypeToFormat(dtype);

    CLTensor a, b, dst;

    // init OpenCL
    CLScheduler::get().default_init();

    // allocate tensors
    a.allocator()->init(TensorInfo(l, n, format));
    b.allocator()->init(TensorInfo(m, l, format));
    dst.allocator()->init(TensorInfo(m, n, format));
    a.allocator()->allocate();
    b.allocator()->allocate();
    dst.allocator()->allocate();
    CLScheduler::get().sync();

    // configure gemm function
    CLGEMM gemm;
    gemm.configure(&a, &b, nullptr, &dst, 1.0, 0.0);

    // run test
    gemm.run();
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < times; i++) {
        gemm.run();
    }
    CLScheduler::get().sync();

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // calcuate gflops
    std::chrono::duration<double> fp_ms = end - begin;
    double cost = fp_ms.count() / times;
    return std::make_pair(cost, 2.0 * n * l * m / (1e9) / cost);
}

int main(int argc, const char **argv)
{
    size_t to_test[][3] = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };

    for (size_t i = 0; i < sizeof(to_test) / sizeof(to_test[0]); i++) {
        int n, l, m;
        double cost, gflops;
        n = to_test[i][0];
        l = to_test[i][1];
        m = to_test[i][2];

        std::tie(cost, gflops) = MeasureGemm(n, l, m, "float");

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "size: " << i << ", " << "cost: " << cost  << ", "
                  << "GFLOPS: " << gflops << std::endl;
    }

    return 0;
}

