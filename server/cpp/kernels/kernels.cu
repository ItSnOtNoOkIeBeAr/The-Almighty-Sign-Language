#include <cmath>

#include "kernels.h"

namespace kern {
namespace {

__constant__ float kMean[3] = {123.675f, 116.28f, 103.53f};  // RGB order, 0..255 units (ImageNet), as in rtmw_ref.MEAN
__constant__ float kStd[3] = {58.395f, 57.12f, 57.375f};

// Solves [x y 1] . [a b c]^T = v for three point pairs (Cramer's rule, double).
void solve3(const double x[3], const double y[3], const double v[3], double abc[3]) {
    auto det = [](double a, double b, double c, double d, double e, double f, double g, double h, double i) {
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    };
    const double d = det(x[0], y[0], 1, x[1], y[1], 1, x[2], y[2], 1);
    abc[0] = det(v[0], y[0], 1, v[1], y[1], 1, v[2], y[2], 1) / d;
    abc[1] = det(x[0], v[0], 1, x[1], v[1], 1, x[2], v[2], 1) / d;
    abc[2] = det(x[0], y[0], v[0], x[1], y[1], v[1], x[2], y[2], v[2]) / d;
}

__global__ void preprocess_kernel(const unsigned char* __restrict__ img, int pitch, int w, int h, float a, float b, float c,
                                  float d, float e, float f, float* __restrict__ out, int ow, int oh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    const float sx = a * x + b * y + c, sy = d * x + e * y + f;
    const float fx0 = floorf(sx), fy0 = floorf(sy);
    const int x0 = (int)fx0, y0 = (int)fy0;
    const float tx = sx - fx0, ty = sy - fy0;
    const float w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
    const bool in_x0 = x0 >= 0 && x0 < w, in_x1 = x0 + 1 >= 0 && x0 + 1 < w;
    const bool in_y0 = y0 >= 0 && y0 < h, in_y1 = y0 + 1 >= 0 && y0 + 1 < h;
    for (int ch = 0; ch < 3; ++ch) {
        const int bc = 2 - ch;  // output channel 0 = R = byte 2 of a BGR pixel
        auto tap = [&](bool ok, int xx, int yy) { return ok ? (float)img[(size_t)yy * pitch + xx * 3 + bc] : 0.0f; };
        const float v = w00 * tap(in_x0 && in_y0, x0, y0) + w10 * tap(in_x1 && in_y0, x0 + 1, y0) +
                        w01 * tap(in_x0 && in_y1, x0, y0 + 1) + w11 * tap(in_x1 && in_y1, x0 + 1, y0 + 1);
        const float v8 = fminf(fmaxf(floorf(v + 0.5f), 0.0f), 255.0f);  // warpAffine gives 8-bit pixels
        out[(size_t)ch * oh * ow + (size_t)y * ow + x] = __fdiv_rn(__fsub_rn(v8, kMean[ch]), kStd[ch]);
    }
}

// One block per keypoint: argmax over the SimCC row of each axis, lowest index wins ties.
__device__ void block_argmax(const float* __restrict__ row, int n, float* best, int* idx, float* sv, int* si) {
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v > bv || (v == bv && i < bi)) bv = v, bi = i;
    }
    sv[threadIdx.x] = bv, si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) sv[threadIdx.x] = ov, si[threadIdx.x] = oi;
        }
        __syncthreads();
    }
    *best = sv[0], *idx = si[0];
    __syncthreads();
}

__global__ void simcc_kernel(const float* __restrict__ sx, const float* __restrict__ sy, int nx, int ny, float in_w, float in_h,
                             float cx, float cy, float scx, float scy, float* __restrict__ out) {
    __shared__ float sv[256];
    __shared__ int si[256];
    const int k = blockIdx.x;
    float mx, my;
    int ix, iy;
    block_argmax(sx + (size_t)k * nx, nx, &mx, &ix, sv, si);
    block_argmax(sy + (size_t)k * ny, ny, &my, &iy, sv, si);
    if (threadIdx.x != 0) return;
    const float score = 0.5f * (mx + my);
    // numpy float32 sequence: kpt / wh * scale + center - scale / 2 (separate roundings, so no fused multiply-add)
    float px = __fdiv_rn((float)ix / 2.0f, in_w), py = __fdiv_rn((float)iy / 2.0f, in_h);
    px = __fsub_rn(__fadd_rn(__fmul_rn(px, scx), cx), scx / 2.0f);
    py = __fsub_rn(__fadd_rn(__fmul_rn(py, scy), cy), scy / 2.0f);
    if (score <= 0) px = py = -1.0f;
    out[k * 3 + 0] = px, out[k * 3 + 1] = py, out[k * 3 + 2] = score;
}

}  // namespace

PoseCrop make_pose_crop(int img_w, int img_h, int in_w, int in_h) {
    // float32 arithmetic exactly as rtmw_ref.pose_crop
    const float cx = (0.0f + img_w) / 2, cy = (0.0f + img_h) / 2;
    float s0 = (float)img_w * 1.25f, s1 = (float)img_h * 1.25f;
    const float aspect = (float)((double)in_w / in_h);
    if (s0 > s1 * aspect) s1 = s0 / aspect;
    else s0 = s1 * aspect;
    // src (frame) points: centre, a point above it, its 90-degree rotation; dst (crop) points likewise
    float src[3][2], dst[3][2];
    src[0][0] = cx, src[0][1] = cy;
    src[1][0] = cx + 0.0f, src[1][1] = cy + s0 * -0.5f;
    src[2][0] = src[1][0] + -(src[0][1] - src[1][1]), src[2][1] = src[1][1] + (src[0][0] - src[1][0]);
    dst[0][0] = in_w * 0.5f, dst[0][1] = in_h * 0.5f;
    dst[1][0] = dst[0][0] + 0.0f, dst[1][1] = dst[0][1] + in_w * -0.5f;
    dst[2][0] = dst[1][0] + -(dst[0][1] - dst[1][1]), dst[2][1] = dst[1][1] + (dst[0][0] - dst[1][0]);
    double dx[3], dy[3], sxv[3], syv[3], ax[3], ay[3];
    for (int i = 0; i < 3; ++i) dx[i] = dst[i][0], dy[i] = dst[i][1], sxv[i] = src[i][0], syv[i] = src[i][1];
    solve3(dx, dy, sxv, ax);  // crop pixel -> frame x
    solve3(dx, dy, syv, ay);  // crop pixel -> frame y
    PoseCrop p;
    for (int i = 0; i < 3; ++i) p.inv[i] = (float)ax[i], p.inv[3 + i] = (float)ay[i];
    p.center[0] = cx, p.center[1] = cy, p.scale[0] = s0, p.scale[1] = s1;
    return p;
}

void preprocess(const unsigned char* img, int pitch, int w, int h, const PoseCrop& c, float* out, int in_w, int in_h,
                cudaStream_t s) {
    const dim3 block(32, 8), grid((in_w + 31) / 32, (in_h + 7) / 8);
    preprocess_kernel<<<grid, block, 0, s>>>(img, pitch, w, h, c.inv[0], c.inv[1], c.inv[2], c.inv[3], c.inv[4], c.inv[5], out,
                                             in_w, in_h);
}

void simcc_decode(const float* simcc_x, const float* simcc_y, int K, int nx, int ny, int in_w, int in_h, const PoseCrop& c,
                  float* out, cudaStream_t s) {
    simcc_kernel<<<K, 256, 0, s>>>(simcc_x, simcc_y, nx, ny, (float)in_w, (float)in_h, c.center[0], c.center[1], c.scale[0],
                                   c.scale[1], out);
}

}  // namespace kern
