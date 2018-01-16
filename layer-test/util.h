#ifndef ARM_COMPUTE_UTIL_H_
#define ARM_COMPUTE_UTIL_H_

#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/core/Types.h"
#include <string>

using namespace arm_compute;

// read data from CLTensor
void ReadTensor(CLTensor &tensor, void *to, size_t size);

// write data to CLTensor
void WriteTensor(CLTensor &tensor, void *from, size_t size);

// transform dtype to format in arm compute
Format DtypeToFormat(std::string);

#endif // ARM_COMPUTE_UTIL_H_

