// nvJPEG decode of one JPEG into a preallocated device buffer (BGR interleaved, tight rows).
#pragma once
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <cstddef>

struct DeviceImage {
    unsigned char* data;
    int width, height, pitch;
};

class JpegDecoder {
public:
    JpegDecoder(int max_w, int max_h, cudaStream_t stream);
    ~JpegDecoder();
    JpegDecoder(const JpegDecoder&) = delete;
    JpegDecoder& operator=(const JpegDecoder&) = delete;
    // Throws DecodeError for anything that is not a decodable JPEG within the size limit. Work is queued on the stream.
    DeviceImage decode(const unsigned char* jpeg, size_t n);

private:
    int max_w_, max_h_;
    cudaStream_t stream_;
    nvjpegHandle_t handle_{};
    nvjpegJpegState_t state_{};
    unsigned char* buf_ = nullptr;
};
