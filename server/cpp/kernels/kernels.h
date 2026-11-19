// CUDA kernels of the keypoint stage. CPU references for both live in tests/test_kernels.cpp; the Python reference they
// follow is training/rtmw_ref.py (pose_crop, simcc_decode).
#pragma once
#include <cuda_runtime.h>

namespace kern {

// Top-down crop of the FULL frame (no detector): bbox [0,0,w,h], padding 1.25, aspect fixed to the model input,
// 3-point affine as in mmpose. src = inv[0..2] . (x, y, 1) is the source x of output pixel (x, y), src_y uses inv[3..5].
struct PoseCrop {
    float inv[6];
    float center[2];
    float scale[2];
};
PoseCrop make_pose_crop(int img_w, int img_h, int in_w, int in_h);

// Fused: affine warp (bilinear, cv2.warpAffine pixel convention: output pixel (x, y) samples the source at the integer-
// aligned point inv.(x, y), taps outside the image are 0) + round to 8 bit + BGR->RGB + (v - mean) / std + HWC->NCHW.
// img: BGR interleaved uint8 on the device (row pitch in bytes). out: float [3, in_h, in_w] on the device.
void preprocess(const unsigned char* img, int pitch, int w, int h, const PoseCrop& crop, float* out, int in_w, int in_h,
                cudaStream_t s);

// SimCC decode: per keypoint argmax of both axes (first maximum wins, like numpy), score = mean of the two peaks,
// mapped back to frame pixels (rtmw_ref.simcc_decode). out: [K, 3] = x, y, score; score <= 0 gives x = y = -1.
void simcc_decode(const float* simcc_x, const float* simcc_y, int K, int nx, int ny, int in_w, int in_h, const PoseCrop& crop,
                  float* out, cudaStream_t s);

}  // namespace kern
