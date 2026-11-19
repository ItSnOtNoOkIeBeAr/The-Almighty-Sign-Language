// Shared helpers: error-checking macros with file/line/error string, and the decode error type.
#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "processor.h"  // DecodeError

[[noreturn]] inline void fail_at(const char* file, int line, const char* what, const std::string& err) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + what + " failed: " + err);
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) fail_at(__FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
    } while (0)

#define TRT_CHECK(cond)                                                           \
    do {                                                                          \
        if (!(cond)) fail_at(__FILE__, __LINE__, #cond, "TensorRT call returned false/null"); \
    } while (0)
