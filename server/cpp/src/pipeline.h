// GPU pipeline: nvJPEG decode -> fused crop/normalise kernel -> RTMW (TensorRT FP16) -> SimCC decode kernel
// -> features (CPU, tiny) -> SignNet (TensorRT FP16) -> softmax -> decision.
// All device and pinned buffers are allocated once in the constructor; a frame allocates nothing on the GPU.
// Two "stages" (profiles) can be loaded: the primary (RTMW-x 384x288) and the small fallback (RTMW-l-m 256x192). The pipeline
// starts on the configured one and drops to the fallback at runtime (tier-1 degrade) when frames stay too slow, VRAM runs low
// or the primary stage throws; it never goes back by itself (no flapping). --mode safe starts on the fallback.
#pragma once
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "config.h"
#include "features.h"
#include "jpeg.h"
#include "processor.h"
#include "trt.h"

// 133 keypoints of one frame: x, y (frame pixels), score
struct Keypoints {
    std::array<float, feat::kKeypoints * 3> v{};
    int width = 0, height = 0;
};

class Pipeline : public IProcessor {
public:
    explicit Pipeline(const Config& cfg);
    ~Pipeline() override;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // One frame of one stream. Thread-safe (frames of all connections share the GPU and take turns).
    // want_keypoints: also fill Result::keypoints (debug view on the client).
    Result process(StreamState& st, const unsigned char* jpeg, size_t n, double ts_ms, bool want_keypoints = false) override;

    // Keypoint stage only (the --extract tool). If blob_out is given, the preprocessed network input is copied there
    // (3 x in_h x in_w floats) so it can be compared with the Python reference.
    Keypoints extract(const unsigned char* jpeg, size_t n, std::vector<float>* blob_out = nullptr);

    StreamState make_state() const override;
    const Config& config() const { return cfg_; }
    std::string active_profile() const;
    void set_inject_delay_ms(int ms) { inject_delay_ms_ = ms; }  // failure injection: makes every frame slower by this much

private:
    struct Stage;  // engines + buffers of one profile
    Keypoints run_keypoints(Stage& s, const DeviceImage& img, StageMs* ms, cudaEvent_t e_start);
    void degrade(const std::string& reason);  // call with gpu_ held

    Config cfg_;
    std::mutex gpu_;
    cudaStream_t stream_{};
    cudaEvent_t ev_[3]{};
    std::unique_ptr<JpegDecoder> jpeg_;
    unsigned char* d_warm_ = nullptr;
    std::unique_ptr<Stage> stages_[2];  // [0] = configured profile, [1] = degrade target (may be missing)
    std::atomic<int> active_{0};
    int slow_frames_ = 0, frames_since_vram_check_ = 0;
    std::atomic<int> inject_delay_ms_{0};
};
