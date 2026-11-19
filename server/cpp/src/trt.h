// Thin TensorRT wrapper: load a serialized engine, bind device pointers by tensor name, enqueueV3 on a stream.
#pragma once
#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

class TrtEngine {
public:
    // build_hint: the exact command that rebuilds the engine, shown if the file is missing or does not deserialize.
    TrtEngine(const std::string& path, const std::string& build_hint);
    void set_input_shape(const char* name, const nvinfer1::Dims& d);
    void bind(const char* name, void* device_ptr);
    nvinfer1::Dims shape(const char* name) const;
    void enqueue(cudaStream_t s);

private:
    struct Deleter { template <class T> void operator()(T* p) const { delete p; } };
    std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, Deleter> ctx_;
};

size_t volume(const nvinfer1::Dims& d);
