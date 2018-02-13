#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "arm_compute/runtime/Scheduler.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>
#include <unistd.h>

using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

std::unique_ptr<ITensorAccessor> dummy() {
    return arm_compute::support::cpp14::make_unique<DummyAccessor>(1);
}

void get_vgg16(Graph *graph, arm_compute::DataType type) {
    *graph << Tensor(TensorInfo(TensorShape(224U, 224U, 3U, 1U), 1, type))
          // Layer 1
          << ConvolutionLayer( 3U, 3U, 64U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 2
          << ConvolutionLayer( 3U, 3U, 64U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 3
          << ConvolutionLayer( 3U, 3U, 128U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 4
          << ConvolutionLayer( 3U, 3U, 128U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 5
          << ConvolutionLayer( 3U, 3U, 256U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 6
          << ConvolutionLayer( 3U, 3U, 256U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 7
          << ConvolutionLayer( 3U, 3U, 256U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 8
          << ConvolutionLayer( 3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 9
          << ConvolutionLayer( 3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 10
          << ConvolutionLayer( 3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 11
          << ConvolutionLayer( 3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 12
          << ConvolutionLayer( 3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 13
          << ConvolutionLayer( 3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 14
          << FullyConnectedLayer( 4096U, dummy(), dummy())
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 15
          << FullyConnectedLayer( 4096U, dummy(), dummy())
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 16
          << FullyConnectedLayer( 1000U, dummy(), dummy())
          // Softmax
          << SoftmaxLayer()
          << Tensor(TensorInfo(TensorShape(1000U), 1, type));
}

BranchLayer get_dwsc_node(const std::string &data_path, std::string &&param_path,
                          unsigned int  conv_filt,
                          PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
{
    std::string total_path = "/cnn_data/mobilenet_v1_model/" + param_path + "_";
    SubGraph    sg;
    sg << DepthwiseConvolutionLayer(
                   3U, 3U, dummy(),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   dwc_pad_stride_info,
                   true)
       << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.001f)
       << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
       << ConvolutionLayer( 1U, 1U, conv_filt, dummy(),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), conv_pad_stride_info)
       << BatchNormalizationLayer( dummy(), dummy(), dummy(), dummy(), 0.001f)
       << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));

    return BranchLayer(std::move(sg));
}

void get_mobilenet(Graph *graph, arm_compute::DataType type) {
    std::string data_path; /* Path to the trainable data */

    *graph << Tensor(TensorInfo(TensorShape(224U, 224U, 3U, 1U), 1, type))
          << ConvolutionLayer( 3U, 3U, 32U, dummy(),
              std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
              PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
          << BatchNormalizationLayer( dummy(), dummy(), dummy(), dummy(), 0.001f)
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
          << get_dwsc_node(data_path, "Conv2d_1", 64, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_2", 128, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_3", 128, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_4", 256, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_5", 256, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_6", 512, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_7", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_8", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_9", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_10", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_11", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_12", 1024, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_13", 1024, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
          << ConvolutionLayer( 1U, 1U, 1000U, dummy(), dummy(), PadStrideInfo(1, 1, 0, 0))
          << ReshapeLayer(TensorShape(1000U))
          << SoftmaxLayer()
          << Tensor(TensorInfo(TensorShape(1000U), 1, type));
}

double measure(Graph *graph, int n_times) {
    arm_compute::CLScheduler::get().default_init();
    graph->run();
    arm_compute::CLScheduler::get().sync();

    auto tbegin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_times; i++) {
        graph->run();
    }
    arm_compute::CLScheduler::get().sync();
    auto tend = std::chrono::high_resolution_clock::now();


    double cost = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
    return cost / n_times;
}

double run_case(std::string backend, std::string model, std::string conv_method, std::string dtype) {
    TargetHint            target_hint;
    ConvolutionMethodHint convolution_hint;
    arm_compute::DataType type;

    if (conv_method == "gemm") {
        convolution_hint = ConvolutionMethodHint::GEMM;
    } else {
        convolution_hint = ConvolutionMethodHint::DIRECT;
    }

    if (backend == "cl") {
        target_hint = TargetHint::OPENCL;
    } else {
        target_hint = TargetHint::NEON;
    }

    if (dtype == "float32") {
        type = DataType::F32;
    } else {
        type = DataType::F16;
    }

    Graph graph;
    graph << target_hint << convolution_hint;

    if (model == "mobilenet")
        get_mobilenet(&graph, type);
    else if (model == "vgg16")
        get_vgg16(&graph, type);
    else
        std::cout << "unknown model" << std::endl;

    int num_warmup, num_test;

    num_warmup = 10;
    num_test   = 60;

    if (model == "mobilenet") { // mobilenet is fast, need more runs for stable measureament
        num_warmup *= 5;
        num_test   *= 5;
    }

    // warm up
    measure(&graph, num_warmup);

    // test
    double cost = measure(&graph, num_test);
    return cost;
}

int main(int argc, const char **argv)
{
    // Check if OpenCL is available and initialize the scheduler
    // Usage 1 : test all
    // Usage 2 : test [cl|neno] [mobilenet|vgg16] [gemm|direct] [float32|float16]

    std::ofstream fout("result-acl.txt", std::ios::app);
    
    if (strcmp(argv[1], "all") == 0) { // test all
        std::string backend[] = {"cl", "neon"};
        std::string model[] = {"vgg16", "mobilenet"};
        std::string conv_method[] = {"gemm", "direct"};
        std::string dtype[] = {"float32", "float16"};

        for (int i = 0; i < sizeof(backend)/sizeof(backend[0]); i++) {
            for (int j = 0; j < sizeof(model)/sizeof(model[0]); j++) {
                for (int k = 0; k < sizeof(conv_method)/sizeof(conv_method[0]); k++) {
                    for (int l = 0; l < sizeof(dtype)/sizeof(dtype[0]); l++) {

                        // skip some test for neon
                        if (backend[i] == "neon" ) {
                            continue;
                            if (conv_method[k] == "direct") // this config is too slow, skip it
                                continue;
                            if (model[j] == "mobilenet")    // too slow, skip it
                                continue;
                            if (dtype[l] == "float16")      // skip the test of fp16 on CPU
                                continue;
                        } else {
                            // ACL does not support FP16 depthwise conv
                            if (model[j] == "mobilenet" && dtype[l] == "float16") 
                                continue;
                        }

                        double cost = run_case(backend[i], model[j], conv_method[k], dtype[l]);

                        std::stringstream ss;

                        std::string back_name;
                        if (backend[i] == "cl")
                            back_name = "mali";
                        else
                            back_name = "neon";

                        ss << "backend: ARMComputeLib-" << back_name << "\tmodel: " << model[j]
                           << "\tconv_method: " << conv_method[k] << "\tdtype: " << dtype[l]
                           << "\tcost: "  << cost;
                        std::cout << ss.str() << std::endl;
                        fout << ss.str() << std::endl;
                        sleep(20);
                    }
                }
            }
        }
    } else { // test single case
        std::string backend = argv[1];
        std::string model = argv[2];
        std::string conv_method = argv[3];
        std::string dtype = argv[4];

        double cost = run_case(backend, model, conv_method, dtype);
        std::stringstream ss;
        ss << "backend: " << backend << "\tmodel: " << model
           << "\tconv_method: " << conv_method << "\tdtype: " << dtype
           << "\tcost: "  << cost;
        std::cout << ss.str() << std::endl;
        fout << ss.str() << std::endl;
    }

    return 0;
}

