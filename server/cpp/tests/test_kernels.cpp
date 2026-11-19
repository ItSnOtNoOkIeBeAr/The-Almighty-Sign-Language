// CUDA kernels against (a) CPU reference loops and (b) the real Python reference (cv2.warpAffine + numpy SimCC decode).
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "common.h"
#include "golden.h"
#include "kernels.h"

namespace {
template <class T>
struct Dev {
    T* p = nullptr;
    size_t n;
    explicit Dev(size_t n_) : n(n_) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    void up(const std::vector<T>& v) { CUDA_CHECK(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down() const {
        std::vector<T> v(n);
        CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

std::vector<float> gpu_preprocess(const std::vector<unsigned char>& img, int w, int h, const kern::PoseCrop& c, int iw, int ih) {
    Dev<unsigned char> d_img(img.size());
    d_img.up(img);
    Dev<float> d_out((size_t)3 * iw * ih);
    kern::preprocess(d_img.p, w * 3, w, h, c, d_out.p, iw, ih, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return d_out.down();
}

// CPU reference of the fused kernel: same sampling convention, straight loops.
std::vector<float> cpu_preprocess(const std::vector<unsigned char>& img, int w, int h, const kern::PoseCrop& c, int iw, int ih) {
    const float mean[3] = {123.675f, 116.28f, 103.53f}, stdv[3] = {58.395f, 57.12f, 57.375f};
    std::vector<float> out((size_t)3 * iw * ih);
    auto px = [&](int x, int y, int bc) { return (x < 0 || y < 0 || x >= w || y >= h) ? 0.0f : (float)img[((size_t)y * w + x) * 3 + bc]; };
    for (int y = 0; y < ih; ++y)
        for (int x = 0; x < iw; ++x) {
            const float sx = c.inv[0] * x + c.inv[1] * y + c.inv[2], sy = c.inv[3] * x + c.inv[4] * y + c.inv[5];
            const int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
            const float tx = sx - x0, ty = sy - y0;
            for (int ch = 0; ch < 3; ++ch) {
                const int bc = 2 - ch;
                const float v = (1 - tx) * (1 - ty) * px(x0, y0, bc) + tx * (1 - ty) * px(x0 + 1, y0, bc) + (1 - tx) * ty * px(x0, y0 + 1, bc) +
                                tx * ty * px(x0 + 1, y0 + 1, bc);
                const float v8 = std::fmin(std::fmax(std::floor(v + 0.5f), 0.0f), 255.0f);
                out[(size_t)ch * ih * iw + (size_t)y * iw + x] = (v8 - mean[ch]) / stdv[ch];
            }
        }
    return out;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}
}  // namespace

TEST(Kernels, PoseCropMatchesPythonGeometry) {
    const auto& g = golden()["preprocess"];
    const kern::PoseCrop c = kern::make_pose_crop(g["w"], g["h"], g["in_w"], g["in_h"]);
    EXPECT_FLOAT_EQ(c.center[0], g["center"][0].get<float>());
    EXPECT_FLOAT_EQ(c.center[1], g["center"][1].get<float>());
    EXPECT_FLOAT_EQ(c.scale[0], g["scale"][0].get<float>());
    EXPECT_FLOAT_EQ(c.scale[1], g["scale"][1].get<float>());
}

TEST(Kernels, PreprocessMatchesCpuReference) {
    // 640x480 live frame -> 288x384 network input, noise image: every pixel counts
    std::mt19937 rng(1);
    std::vector<unsigned char> img(640 * 480 * 3);
    for (auto& v : img) v = (unsigned char)(rng() & 255);
    const kern::PoseCrop c = kern::make_pose_crop(640, 480, 288, 384);
    const auto gpu = gpu_preprocess(img, 640, 480, c, 288, 384), cpu = cpu_preprocess(img, 640, 480, c, 288, 384);
    size_t exact = 0;
    for (size_t i = 0; i < gpu.size(); ++i) exact += gpu[i] == cpu[i];
    // GPU fused multiply-add can move a value across a .5 rounding boundary by one grey level (1/57 after normalising), rarely
    EXPECT_LE(max_abs_diff(gpu, cpu), 1.0 / 57.0 + 1e-5);
    EXPECT_GT((double)exact / gpu.size(), 0.999);
}

TEST(Kernels, PreprocessMatchesOpenCvWarpAffine) {
    const auto& g = golden()["preprocess"];
    const int w = g["w"], h = g["h"], iw = g["in_w"], ih = g["in_h"];
    const auto img = g["img"].get<std::vector<unsigned char>>();
    const auto want = g["blob"].get<std::vector<float>>();
    const auto got = gpu_preprocess(img, w, h, kern::make_pose_crop(w, h, iw, ih), iw, ih);
    // cv2 interpolates in 1/32-pixel fixed point, so it can differ from exact bilinear by ~1 grey level (1/57 normalised)
    EXPECT_LE(max_abs_diff(got, want), 2.0 / 57.0) << "kernel disagrees with cv2.warpAffine + normalisation";
}

TEST(Kernels, OutsidePixelsAreZeroBlack) {
    // crop that reaches far outside a tiny image: taps beyond the border contribute 0 (black), like warpAffine's default border
    std::vector<unsigned char> img(8 * 8 * 3, 255);
    const auto got = gpu_preprocess(img, 8, 8, kern::make_pose_crop(8, 8, 24, 32), 24, 32);
    const float black = (0 - 123.675f) / 58.395f;
    EXPECT_FLOAT_EQ(got[0], black);  // top-left corner of the R plane lies outside the image
    const float white = (255 - 123.675f) / 58.395f;
    EXPECT_FLOAT_EQ(got[16 * 24 + 12], white);  // centre pixel is inside
}

TEST(Kernels, SimccDecodeMatchesPython) {
    const auto& g = golden()["simcc"];
    const auto& pp = golden()["preprocess"];
    const int K = g["k"], nx = g["nx"], ny = g["ny"];
    Dev<float> sx(K * nx), sy(K * ny), out(K * 3);
    sx.up(g["sx"].get<std::vector<float>>()), sy.up(g["sy"].get<std::vector<float>>());
    kern::simcc_decode(sx.p, sy.p, K, nx, ny, pp["in_w"], pp["in_h"], kern::make_pose_crop(pp["w"], pp["h"], pp["in_w"], pp["in_h"]), out.p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto got = out.down();
    const auto kpt = g["kpt"].get<std::vector<float>>(), score = g["score"].get<std::vector<float>>();
    for (int k = 0; k < K; ++k) {
        EXPECT_NEAR(got[3 * k], kpt[2 * k], 1e-3) << "keypoint " << k;
        EXPECT_NEAR(got[3 * k + 1], kpt[2 * k + 1], 1e-3) << "keypoint " << k;
        EXPECT_NEAR(got[3 * k + 2], score[k], 1e-4) << "keypoint " << k;
    }
}

TEST(Kernels, SimccDecodeFullSizeMatchesCpuReference) {
    // real model size (133 keypoints, 576 + 768 bins), random rows incl. exact ties
    const int K = 133, nx = 576, ny = 768, iw = 288, ih = 384;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> hx(K * nx), hy(K * ny);
    for (auto& v : hx) v = std::round(nd(rng) * 4) / 4;  // quantised: plenty of ties
    for (auto& v : hy) v = std::round(nd(rng) * 4) / 4;
    const kern::PoseCrop c = kern::make_pose_crop(640, 480, iw, ih);
    Dev<float> sx(hx.size()), sy(hy.size()), out(K * 3);
    sx.up(hx), sy.up(hy);
    kern::simcc_decode(sx.p, sy.p, K, nx, ny, iw, ih, c, out.p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto got = out.down();
    for (int k = 0; k < K; ++k) {
        int ix = 0, iy = 0;
        for (int i = 1; i < nx; ++i) if (hx[k * nx + i] > hx[k * nx + ix]) ix = i;  // strict >: first maximum wins
        for (int i = 1; i < ny; ++i) if (hy[k * ny + i] > hy[k * ny + iy]) iy = i;
        const float score = 0.5f * (hx[k * nx + ix] + hy[k * ny + iy]);
        float px = ((float)ix / 2.0f) / iw * c.scale[0] + c.center[0] - c.scale[0] / 2.0f;
        float py = ((float)iy / 2.0f) / ih * c.scale[1] + c.center[1] - c.scale[1] / 2.0f;
        if (score <= 0) px = py = -1;
        EXPECT_NEAR(got[3 * k], px, 1e-3) << k;
        EXPECT_NEAR(got[3 * k + 1], py, 1e-3) << k;
        EXPECT_FLOAT_EQ(got[3 * k + 2], score) << k;
    }
}
