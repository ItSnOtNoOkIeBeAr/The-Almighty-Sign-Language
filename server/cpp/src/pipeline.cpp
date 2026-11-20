#include "pipeline.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include "../kernels/kernels.h"
#include "common.h"

namespace {
double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
double event_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}
double free_vram_gb() {
    size_t f = 0, t = 0;
    CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return f / 1073741824.0;
}
}  // namespace

// Engines and every buffer of one profile, allocated once.
struct Pipeline::Stage {
    ProfileCfg cfg;
    std::unique_ptr<TrtEngine> kp, temporal;
    float *d_input = nullptr, *d_sx = nullptr, *d_sy = nullptr, *d_kpts = nullptr, *d_x = nullptr, *d_logits = nullptr;
    float *h_kpts = nullptr, *h_x = nullptr, *h_logits = nullptr;  // pinned
    int nx = 0, ny = 0, n_classes = 0;

    Stage(const ProfileCfg& pr, size_t n_labels) : cfg(pr) {
        kp = std::make_unique<TrtEngine>(pr.keypoint_engine, "powershell -File server\\scripts\\build_engines.ps1 -Profile " + pr.name);
        temporal = std::make_unique<TrtEngine>(pr.temporal_engine, "training\\.venv\\Scripts\\python training\\export_onnx.py --model <model> --out server/models/temporal_" + pr.name);
        kp->set_input_shape("input", nvinfer1::Dims4{1, 3, pr.in_h, pr.in_w});
        const nvinfer1::Dims sx = kp->shape("simcc_x"), sy = kp->shape("simcc_y");
        if (sx.nbDims != 3 || sx.d[1] != feat::kKeypoints || sy.d[1] != feat::kKeypoints)
            throw std::runtime_error("keypoint engine outputs are not [1,133,N]");
        nx = (int)sx.d[2], ny = (int)sy.d[2];
        if (nx != pr.in_w * 2 || ny != pr.in_h * 2) throw std::runtime_error("SimCC length does not match 2x the model input size");
        const nvinfer1::Dims lo = temporal->shape("logits");
        n_classes = (int)lo.d[lo.nbDims - 1];
        if (n_classes != (int)n_labels)
            throw std::runtime_error("temporal engine has " + std::to_string(n_classes) + " classes, labels.json has " + std::to_string(n_labels));
        CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * 3 * pr.in_h * pr.in_w));
        CUDA_CHECK(cudaMalloc(&d_sx, sizeof(float) * feat::kKeypoints * nx));
        CUDA_CHECK(cudaMalloc(&d_sy, sizeof(float) * feat::kKeypoints * ny));
        CUDA_CHECK(cudaMalloc(&d_kpts, sizeof(float) * feat::kKeypoints * 3));
        CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * feat::kWindow * feat::kDim));
        CUDA_CHECK(cudaMalloc(&d_logits, sizeof(float) * n_classes));
        CUDA_CHECK(cudaMallocHost(&h_kpts, sizeof(float) * feat::kKeypoints * 3));
        CUDA_CHECK(cudaMallocHost(&h_x, sizeof(float) * feat::kWindow * feat::kDim));
        CUDA_CHECK(cudaMallocHost(&h_logits, sizeof(float) * n_classes));
        kp->bind("input", d_input), kp->bind("simcc_x", d_sx), kp->bind("simcc_y", d_sy);
        temporal->bind("x", d_x), temporal->bind("logits", d_logits);
    }
    ~Stage() {
        for (void* p : {(void*)d_input, (void*)d_sx, (void*)d_sy, (void*)d_kpts, (void*)d_x, (void*)d_logits}) cudaFree(p);
        for (void* p : {(void*)h_kpts, (void*)h_x, (void*)h_logits}) cudaFreeHost(p);
    }
};

Pipeline::Pipeline(const Config& cfg) : cfg_(cfg) {
    // 1. GPU and compute capability: the kernels and engines are built for Blackwell (sm_120)
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    spdlog::info("GPU: {} (sm_{}{}), {:.1f} GB", p.name, p.major, p.minor, p.totalGlobalMem / 1073741824.0);
    if (p.major != 12) throw std::runtime_error("this build targets sm_120 (Blackwell); found sm_" + std::to_string(p.major) + std::to_string(p.minor));
    // 2. VRAM budget
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    spdlog::info("VRAM free {:.1f} GB of {:.1f} GB (need >= {:.1f} GB free, own cap {:.1f} GB)", free_b / 1073741824.0,
                 total_b / 1073741824.0, cfg.vram_min_free_gb, cfg.vram_max_use_gb);
    if (free_b / 1073741824.0 < cfg.vram_min_free_gb)
        throw std::runtime_error("not enough free VRAM: " + std::to_string(free_b / 1073741824.0) + " GB free, need " +
                                 std::to_string(cfg.vram_min_free_gb) + " GB. Close other GPU programs and restart.");
    const size_t used_before = total_b - free_b;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    for (auto& e : ev_) CUDA_CHECK(cudaEventCreate(&e));

    // 3. engines + 4. preallocation. The configured profile must load; the degrade target is optional.
    stages_[0] = std::make_unique<Stage>(cfg.profile, cfg.labels.size());
    if (cfg.alt && cfg.degrade.enabled) {
        try {
            stages_[1] = std::make_unique<Stage>(*cfg.alt, cfg.labels.size());
        } catch (const std::exception& e) {
            spdlog::warn("degrade target '{}' is not available, running without tier-1 degrade: {}", cfg.alt->name, e.what());
        }
    }
    jpeg_ = std::make_unique<JpegDecoder>(cfg.max_frame_w, cfg.max_frame_h, stream_);
    CUDA_CHECK(cudaMalloc(&d_warm_, 640 * 480 * 3));
    CUDA_CHECK(cudaMemset(d_warm_, 0, 640 * 480 * 3));

    // warm-up (first enqueue pays one-time costs), both stages so a later degrade is not slow
    for (auto& st : stages_) {
        if (!st) continue;
        for (int i = 0; i < 3; ++i) {
            run_keypoints(*st, {d_warm_, 640, 480, 640 * 3}, nullptr, ev_[0]);
            CUDA_CHECK(cudaMemsetAsync(st->d_x, 0, sizeof(float) * feat::kWindow * feat::kDim, stream_));
            st->temporal->enqueue(stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    spdlog::info("profile {}{}: engines loaded, buffers preallocated, VRAM in use by this server ~{:.2f} GB", cfg.profile.name,
                 stages_[1] ? " (+ " + cfg.alt->name + " as degrade target)" : "", ((total_b - free_b) - used_before) / 1073741824.0);
}

Pipeline::~Pipeline() {
    cudaStreamSynchronize(stream_);
    stages_[0].reset(), stages_[1].reset();
    cudaFree(d_warm_);
    jpeg_.reset();
    for (auto& e : ev_) cudaEventDestroy(e);
    cudaStreamDestroy(stream_);
}

StreamState Pipeline::make_state() const {
    StreamState st(cfg_.labels, stages_[active_]->cfg.decision, cfg_.step_ms);
    st.profile_id = active_;
    return st;
}

std::string Pipeline::active_profile() const { return stages_[active_]->cfg.name; }

void Pipeline::degrade(const std::string& reason) {
    if (active_ != 0 || !stages_[1]) return;
    spdlog::warn("TIER-1 DEGRADE: {} -> {} ({}). Streams restart on the fallback keypoint model; no automatic way back (restart the server).",
                 stages_[0]->cfg.name, stages_[1]->cfg.name, reason);
    active_ = 1;
    slow_frames_ = 0;
}

Keypoints Pipeline::run_keypoints(Stage& s, const DeviceImage& img, StageMs* ms, cudaEvent_t e_start) {
    const auto& pr = s.cfg;
    const kern::PoseCrop crop = kern::make_pose_crop(img.width, img.height, pr.in_w, pr.in_h);
    kern::preprocess(img.data, img.pitch, img.width, img.height, crop, s.d_input, pr.in_w, pr.in_h, stream_);
    s.kp->enqueue(stream_);
    kern::simcc_decode(s.d_sx, s.d_sy, feat::kKeypoints, s.nx, s.ny, pr.in_w, pr.in_h, crop, s.d_kpts, stream_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(s.h_kpts, s.d_kpts, sizeof(float) * feat::kKeypoints * 3, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(ev_[2], stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));  // the one sync per stage: the CPU needs the keypoints
    if (ms) ms->keypoints = event_ms(ev_[1], ev_[2]), ms->decode = event_ms(e_start, ev_[1]);
    Keypoints k;
    std::copy(s.h_kpts, s.h_kpts + feat::kKeypoints * 3, k.v.begin());
    k.width = img.width, k.height = img.height;
    return k;
}

Keypoints Pipeline::extract(const unsigned char* jpeg, size_t n, std::vector<float>* blob_out) {
    std::lock_guard<std::mutex> lk(gpu_);
    Stage& s = *stages_[active_];
    const DeviceImage img = jpeg_->decode(jpeg, n);
    CUDA_CHECK(cudaEventRecord(ev_[1], stream_));
    Keypoints k = run_keypoints(s, img, nullptr, ev_[0]);
    if (blob_out) {
        blob_out->resize((size_t)3 * s.cfg.in_h * s.cfg.in_w);
        CUDA_CHECK(cudaMemcpy(blob_out->data(), s.d_input, blob_out->size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return k;
}

void Pipeline::open_diag(const std::string& path) {
    diag_.open(path, std::ios::out | std::ios::trunc);
    if (!diag_) throw std::runtime_error("cannot open diagnostic log " + path);
    diag_ << "ts_ms,profile";
    for (const auto& l : cfg_.labels) diag_ << ",p_" << l;
    diag_ << ",top1,top2,top3,label,conf,state,committed,hand_L_loose,hand_R_loose,hand_L_strict,hand_R_strict,pts_L,pts_R,thr\n";
    spdlog::info("diagnostic log (numbers only, no images): {}", path);
}

// One CSV line per 10 fps decision tick. pts_L / pts_R = how many of the 21 hand points score above the loose threshold (a hand needs >= 15 to count).
void Pipeline::write_diag(double ts_ms, const Stage& s, const std::vector<double>& probs, const feat::FrameRow& row, const DecisionOut& d) {
    std::vector<size_t> order(probs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
    const int n_l = row.hand_pts[0], n_r = row.hand_pts[1];
    char buf[64];
    diag_ << (long long)ts_ms << ',' << s.cfg.name;
    for (double p : probs) diag_ << ',' << (std::snprintf(buf, sizeof buf, "%.4f", p), buf);
    for (size_t k = 0; k < 3 && k < order.size(); ++k) {
        std::snprintf(buf, sizeof buf, "%s:%.3f", cfg_.labels[order[k]].c_str(), probs[order[k]]);
        diag_ << ',' << buf;
    }
    diag_ << ',' << d.label << ',' << d.confidence << ',' << d.state << ',' << (d.committed ? *d.committed : "") << ',' << row.flags[0] << ',' << row.flags[1]
          << ',' << row.flags[3] << ',' << row.flags[4] << ',' << n_l << ',' << n_r << ',' << s.cfg.keypoint_thr << '\n';
    diag_.flush();
}

Result Pipeline::process(StreamState& st, const unsigned char* jpeg, size_t n, double ts_ms, bool want_keypoints) {
    std::lock_guard<std::mutex> lk(gpu_);
    const auto t0 = std::chrono::steady_clock::now();
    if (st.profile_id != active_) {  // the pipeline degraded since this stream's window was filled: restart it on the new profile
        st.set_decision(stages_[active_]->cfg.decision);
        st.profile_id = active_;
    }
    Stage& s = *stages_[active_];
    Result r;
    r.profile = s.cfg.name;
    try {
        CUDA_CHECK(cudaEventRecord(ev_[0], stream_));
        const DeviceImage img = jpeg_->decode(jpeg, n);  // throws DecodeError
        CUDA_CHECK(cudaEventRecord(ev_[1], stream_));
        const Keypoints k = run_keypoints(s, img, &r.ms, ev_[0]);

        if (want_keypoints) {  // body (0-12: nose, eyes, ears, shoulders, elbows, wrists, hips) and both hands (91-132)
            r.kp_w = k.width, r.kp_h = k.height, r.kp_thr = s.cfg.keypoint_thr;
            for (int i = 0; i < feat::kKeypoints; ++i)
                if (i <= 12 || i >= 91) r.keypoints.insert(r.keypoints.end(), {(float)i, k.v[3 * i], k.v[3 * i + 1], k.v[3 * i + 2]});
        }
        float xy[feat::kKeypoints * 2], score[feat::kKeypoints];
        for (int i = 0; i < feat::kKeypoints; ++i) xy[2 * i] = k.v[3 * i], xy[2 * i + 1] = k.v[3 * i + 1], score[i] = k.v[3 * i + 2];
        feat::FrameRow row = feat::frame_row(xy, score, s.cfg.keypoint_thr, s.cfg.hand_present_thr, k.width, k.height);
        for (int i = 91; i < 112; ++i) row.hand_pts[0] += score[i] >= s.cfg.keypoint_thr;
        for (int i = 112; i < 133; ++i) row.hand_pts[1] += score[i] >= s.cfg.keypoint_thr;
        st.push(ts_ms, row);

        // 10 fps recognition clock: the temporal model and the decision run once per 100 ms tick, on the frame closest to it
        const auto t2 = std::chrono::steady_clock::now();
        double t_tick = 0;
        bool run_model = false;
        while (st.pop_tick(&t_tick, &run_model)) {
            if (!run_model) continue;  // a gap in the stream: the missed ticks only fill the window
            const feat::FrameRow* win[feat::kWindow];
            st.window(win);
            feat::window_features(win, s.h_x);
            CUDA_CHECK(cudaMemcpyAsync(s.d_x, s.h_x, sizeof(float) * feat::kWindow * feat::kDim, cudaMemcpyHostToDevice, stream_));
            s.temporal->enqueue(stream_);
            CUDA_CHECK(cudaMemcpyAsync(s.h_logits, s.d_logits, sizeof(float) * s.n_classes, cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::vector<double> probs(s.n_classes);
            const float mx = *std::max_element(s.h_logits, s.h_logits + s.n_classes);
            double sum = 0;
            for (int i = 0; i < s.n_classes; ++i) sum += (probs[i] = std::exp((double)s.h_logits[i] - mx));
            for (auto& p : probs) p /= sum;
            const feat::FrameRow& tick_row = *win[feat::kWindow - 1];
            const DecisionOut d = st.decision.step(probs, tick_row.flags[3] || tick_row.flags[4], t_tick);  // strict presence drives the idle gate
            st.record(d);
            if (diag_.is_open()) write_diag(t_tick, s, probs, tick_row, d);
        }
        r.ms.temporal = ms_since(t2);  // 0 for frames that are not a tick
        const DecisionOut d = st.report();
        r.label = d.label, r.confidence = d.confidence, r.state = d.state, r.committed = d.committed;
    } catch (const DecodeError&) {
        throw;  // a bad frame says nothing about the health of the GPU stage
    } catch (const std::exception& e) {
        if (active_ == 0 && stages_[1] && cfg_.degrade.enabled) degrade(std::string("primary stage failed: ") + e.what());
        throw;  // this frame gets an "internal" error reply; the next ones run on the fallback
    }
    if (const int delay = inject_delay_ms_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(delay));  // failure injection only
    r.ms.total = ms_since(t0);

    // tier-1 degrade triggers: sustained slow frames, or low free VRAM
    if (active_ == 0 && stages_[1] && cfg_.degrade.enabled) {
        slow_frames_ = r.ms.total > cfg_.degrade.latency_ms ? slow_frames_ + 1 : 0;
        if (slow_frames_ >= cfg_.degrade.frames)
            degrade("server time > " + std::to_string((int)cfg_.degrade.latency_ms) + " ms for " + std::to_string(cfg_.degrade.frames) + " frames in a row");
        else if (++frames_since_vram_check_ >= 100) {
            frames_since_vram_check_ = 0;
            const double f = free_vram_gb();
            if (f < cfg_.degrade.min_free_vram_gb) degrade("free VRAM " + std::to_string(f) + " GB < " + std::to_string(cfg_.degrade.min_free_vram_gb) + " GB");
        }
    }
    return r;
}
