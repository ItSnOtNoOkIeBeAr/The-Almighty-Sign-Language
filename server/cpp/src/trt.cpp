#include "trt.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <iterator>

#include "common.h"

namespace {
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR) spdlog::error("[TensorRT] {}", msg);
        else if (s == Severity::kWARNING) spdlog::warn("[TensorRT] {}", msg);
    }
};
TrtLogger g_logger;
}  // namespace

size_t volume(const nvinfer1::Dims& d) {
    size_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) n *= static_cast<size_t>(d.d[i]);
    return n;
}

TrtEngine::TrtEngine(const std::string& path, const std::string& build_hint) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("engine file not found: " + path + "\n  build it with: " + build_hint);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    TRT_CHECK(runtime_);
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_)
        throw std::runtime_error("engine '" + path + "' did not load (built for a different TensorRT version or GPU?)\n  rebuild: " +
                                 build_hint);
    ctx_.reset(engine_->createExecutionContext());
    TRT_CHECK(ctx_);
}

void TrtEngine::set_input_shape(const char* name, const nvinfer1::Dims& d) { TRT_CHECK(ctx_->setInputShape(name, d)); }
void TrtEngine::bind(const char* name, void* p) { TRT_CHECK(ctx_->setTensorAddress(name, p)); }
nvinfer1::Dims TrtEngine::shape(const char* name) const { return ctx_->getTensorShape(name); }
void TrtEngine::enqueue(cudaStream_t s) { TRT_CHECK(ctx_->enqueueV3(s)); }
