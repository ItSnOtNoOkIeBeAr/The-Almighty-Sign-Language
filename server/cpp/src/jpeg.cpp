#include "jpeg.h"

#include <string>

#include "common.h"

#define NVJPEG_CHECK(call)                                                               \
    do {                                                                                 \
        nvjpegStatus_t s_ = (call);                                                      \
        if (s_ != NVJPEG_STATUS_SUCCESS) fail_at(__FILE__, __LINE__, #call, "nvjpeg status " + std::to_string((int)s_)); \
    } while (0)

JpegDecoder::JpegDecoder(int max_w, int max_h, cudaStream_t stream) : max_w_(max_w), max_h_(max_h), stream_(stream) {
    NVJPEG_CHECK(nvjpegCreateSimple(&handle_));
    NVJPEG_CHECK(nvjpegJpegStateCreate(handle_, &state_));
    CUDA_CHECK(cudaMalloc(&buf_, (size_t)max_w * max_h * 3));
}

JpegDecoder::~JpegDecoder() {
    cudaFree(buf_);
    if (state_) nvjpegJpegStateDestroy(state_);
    if (handle_) nvjpegDestroy(handle_);
}

DeviceImage JpegDecoder::decode(const unsigned char* jpeg, size_t n) {
    int comps = 0, w[NVJPEG_MAX_COMPONENT] = {}, h[NVJPEG_MAX_COMPONENT] = {};
    nvjpegChromaSubsampling_t css;
    if (nvjpegGetImageInfo(handle_, jpeg, n, &comps, &css, w, h) != NVJPEG_STATUS_SUCCESS || w[0] <= 0 || h[0] <= 0)
        throw DecodeError("not a decodable JPEG");
    if (w[0] > max_w_ || h[0] > max_h_)
        throw DecodeError("image " + std::to_string(w[0]) + "x" + std::to_string(h[0]) + " exceeds the " + std::to_string(max_w_) + "x" +
                          std::to_string(max_h_) + " limit");
    nvjpegImage_t out{};
    out.channel[0] = buf_;
    out.pitch[0] = (unsigned)w[0] * 3;
    if (nvjpegDecode(handle_, state_, jpeg, n, NVJPEG_OUTPUT_BGRI, &out, stream_) != NVJPEG_STATUS_SUCCESS)
        throw DecodeError("not a decodable JPEG");
    return {buf_, w[0], h[0], w[0] * 3};
}
