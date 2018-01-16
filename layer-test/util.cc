#include "util.h"

// read data from CLTensor
void ReadTensor(const CLTensor *tensor, void *to, size_t size) {
    cl::CommandQueue &queue = CLScheduler::get().queue();
    queue.enqueueReadBuffer(tensor->cl_buffer(), true, 0, size, to);
    queue.finish();
}

// write data to CLTensor
void WriteTensor(CLTensor *tensor, const void *from, size_t size) {
    cl::CommandQueue &queue = CLScheduler::get().queue();
    queue.enqueueWriteBuffer(tensor->cl_buffer(), true, 0, size, from);
    queue.finish();
}

// transform dtype to format in arm compute
Format DtypeToFormat(std::string dtype) {
    if (dtype == "float" || dtype == "float32")
        return Format::F32;
    else if (dtype == "float16")
        return Format::F16;
    else {
        std::cerr << "Unsupported type: " << dtype << std::endl;
        exit(-1);
    }
}
