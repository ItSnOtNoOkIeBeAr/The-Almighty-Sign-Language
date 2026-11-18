# BUILD_SPEC: The Almighty Sign Language

You are building a real-time sign language recognition system on this Windows PC. Read this whole file before doing anything. Then follow the phases in order. This file is the source of truth. If something here is wrong or impossible, say so and propose a fix. Do not silently work around it.

---

## 0. Working rules (non-negotiable)

1. **Verify, never assume.** Every phase ends with a gate: you run the build and tests, show the real output, and only then mark the phase done. Never claim something works without running it.
2. **No silent stubs.** If you write a placeholder, label it `// PLACEHOLDER:` with the reason, and list it in `STATUS.md`. No fake test passes, no tests that only assert `true`, no disabled tests to get green.
3. **Stop and ask** when you need something only the user can do or decide: downloads behind a login (TensorRT), the tunnel hostname, choosing a keypoint model, recording data, anything costly to undo. Ask one clear question with your recommendation.
4. **Fix root causes.** When a build or test fails, find the cause. Do not loosen tolerances, catch-and-ignore errors, or add `-allow-unsupported-compiler`-style flags without explaining why and getting approval.
5. **Keep records.** Maintain:
   - `CLAUDE.md` (repo root): a short summary of these rules, the current phase, and how to build and run. Keep it under ~80 lines so it stays useful every session.
   - `STATUS.md`: per phase, what is verified (with the command used), what is not, known issues, and placeholders.
   - `DECISIONS.md`: every non-obvious choice (library, model, version, tolerance) with a one-line reason.
6. **Git:** commit at the end of each passed gate with a clear message. Never push unless asked. Never commit secrets, datasets, recorded clips, `.onnx`, `.engine`, or build folders.
7. **Secrets:** never ask for, print, store, or commit the Cloudflare tunnel token. The app's own shared secret comes from the `SIGNLANG_TOKEN` environment variable only.
8. **Windows paths:** the user's home folder contains a space (`C:\Users\Matthew Dee`). Put the repo at `C:\dev\almighty-sign-language` to avoid CUDA/CMake path issues, and quote every path in scripts.
9. **Small, readable code.** Prefer clarity over cleverness except in the CUDA hot path, where performance rules (section 7) apply. Comment the "why", not the "what".

---

## 1. Project summary

- **What it does:** a laptop webcam streams video over a Cloudflare Tunnel to this PC. The PC recognizes signs on the GPU and sends back the label, which the laptop shows as an overlay.
- **Sign language: ASL** (decided 2026-11-18). Training data is public only (section 6).
- **Vocabulary (7 classes, approved 2026-11-18; replaces the original 8 whole-phrase classes):**
  `0 none, 1 hello, 2 what, 3 how, 4 why, 5 where, 6 you`.
  - `none` (index 0) is **idle + other**: no sign, or any sign that is not one of the six words. Public data gives it the OTHER class (150 random other glosses) plus hands-down windows from the start and end of all clips.
  - `hello` also covers "hi": ASL Citizen has no separate HI gloss and ASL uses one sign family for both.
  - **Sentences are joined from words, not classified.** `how_are_you` = `how` then `you`; `where_are_you` = `where` then `you` or `you` then `where`, within 1500 ms (section 8). The classifier only knows single words.
  Labels live in `protocol/labels.json`, and every component reads them from there.
- **Primary runtime:** C++/CUDA only, with no Python at runtime and no Python-vs-CUDA benchmark anywhere.
- **Python is allowed only for:** offline training and export, test and parity scripts, the laptop client, and the minimal fallback server.

## 2. Verified environment (this PC)

| Item | Version |
|---|---|
| GPU | RTX 5060 Ti 16 GB, Blackwell, compute capability **sm_120** |
| Driver | 617.42 (supports CUDA 13.4) |
| CUDA toolkit | 13.2 (`nvcc -arch=sm_120` hello-kernel verified) |
| Host compiler | MSVC 19.42 (VS 2022 Community 17.12) |
| CMake | 3.29.5 (bundled with VS) |
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128, `torch.cuda.is_available() == True` |
| TensorRT | **not installed**: Phase 1 |
| vcpkg | **not installed**: Phase 1 |

`cl` is only on PATH inside the VS developer environment. If `cl` is missing in your terminal, tell the user to close VS Code and reopen it from **x64 Native Tools Command Prompt for VS 2022** with `code C:\dev\almighty-sign-language`.

The laptop has **no GPU**. Everything in `client/` must run on a CPU-only Windows laptop with plain `pip` packages.

## 3. Architecture

```
LAPTOP (Python, CPU only)
  webcam -> resize (default 1280x720, configurable) -> JPEG (quality ~80) @ 15 FPS
  -> WebSocket (wss://<TUNNEL_HOSTNAME>/ws) via Cloudflare Tunnel
                         |
PC: signlang_server.exe (C++/CUDA)  [primary]
  WebSocket recv (bytes) -> latest-frame-only slot
  -> nvJPEG decode (GPU)
  -> person/hand detector (TensorRT FP16) -> box decode/select on GPU
  -> fused crop+resize+normalize kernel -> keypoint model (TensorRT FP16)
  -> keypoint decode + normalize kernel -> 45-frame ring buffer (GPU)
  -> temporal model (TensorRT FP16) -> softmax + EMA smoothing kernel
  -> copy back ~8 floats -> decision state machine (CPU, tiny) -> JSON reply
                         |
PC: fallback_server.py (Python, FastAPI + onnxruntime-gpu) [safety net only]
  same protocol, same ONNX models, no custom kernels
```

The exact number of models in the keypoint stage is decided in Phase 3. Fewest models wins.

## 4. Repository layout

```
almighty-sign-language/
  CLAUDE.md  README.md  governance/{BUILD_SPEC,STATUS,DECISIONS}.md  .gitignore
  protocol/
    PROTOCOL.md          wire format (section 5), the contract for all components
    labels.json
    conformance_test.py  runs against ANY server URL, must pass on both servers
  server/
    cpp/
      CMakeLists.txt  CMakePresets.json  vcpkg.json
      src/        main, ws_server, pipeline, nvjpeg_decoder, trt_engine, decision, config, memory_budget
      kernels/    preprocess.cu, detect_post.cu, keypoint_decode.cu, ringbuffer.cu, postprocess.cu
      tests/      GoogleTest: CPU references + kernel tests + decision tests
    fallback/     fallback_server.py, requirements.txt
    models/       *.onnx, *.engine        (gitignored)
    config/       server.json (thresholds, window, paths, VRAM budget)
    scripts/      build_engines.ps1, start_server.ps1, start_fallback.ps1, run_tests.ps1, sanitize.ps1
  client/         client.py, recorder.py, config.json, requirements.txt (CPU-only packages)
  training/       extract_keypoints.py, dataset.py, train.py, export_onnx.py,
                  make_placeholder_model.py, parity_check.py, requirements.txt
  data/           recorded clips (gitignored)
  docs/           diagrams, report notes
```

## 5. Protocol (write `protocol/PROTOCOL.md` exactly from this)

- **Endpoint:** `GET /ws` WebSocket upgrade. Auth: header `Authorization: Bearer <SIGNLANG_TOKEN>`. Reject with HTTP 401 before upgrading if it is missing or wrong. `GET /health` returns `{"ok":true,"server":"cpp"|"fallback","version":"..."}` without auth.
- **Client to server: binary messages**, little-endian:
  ```
  offset 0  uint32 magic = 0x534C4E47 ("SLNG")
  offset 4  uint16 version = 1
  offset 6  uint16 flags (bit0 = end_of_stream)
  offset 8  uint32 frame_id (monotonic)
  offset 12 uint64 client_ts_us
  offset 20 ... JPEG bytes (rest of message)
  ```
  Max message size 2 MB. Anything malformed gets an error reply and is dropped. The connection stays open.
- **Server to client: text messages**, JSON, one per processed frame:
  ```json
  {"type":"result","frame_id":123,"label":"hello","confidence":0.94,
   "state":"idle|signing|committed","committed_label":"hello"|null,
   "server_ms":{"decode":1.2,"keypoints":6.1,"temporal":0.8,"total":9.0},
   "dropped_frames":2}
  ```
  Also `{"type":"error","code":"bad_magic|too_large|decode_failed|...","detail":"..."}`.
- Frames arriving while one is being processed **replace** the waiting frame (latest-only). Never queue more than one.
- `conformance_test.py --url ws://localhost:8765/ws --token ...` checks auth rejection, health, a valid frame round trip, malformed input handling, oversized input, rapid-fire frames (latest-only behaviour), and reconnect. Both servers must pass it.

## 6. Data and recording format

**Data policy (decided 2026-11-18):**
- **Public data only.** Training uses ASL Citizen (Microsoft Research; isolated ASL signs, 83,399 clips, 2,731 glosses, 52 signers, official signer-disjoint train/val/test split). Nobody records training data. The recorder below stays as a tool for test clips; the 5 laptop sample clips of 2026-10-27 were deleted (they showed no signing).
- **Domain matching.** The data is crowd-sourced webcam video, our runtime input is the laptop stream. Every dataset clip is converted to the live format (640x480, JPEG q70, 10 fps) with the same `prepare_frame` + `encode_jpeg` as `client/common.py` before keypoint extraction (aspect ratio padded, never squashed), and training augments lighting, scale and background. Phase 3 and later evaluation use clips in this converted form.
- **Live-check mode.** Because no own data is used for training, the live webcam is a sanity check only: a view-only mode (`client.py --live-check`, planned, not yet written) shows keypoints and predictions and saves nothing. It never produces training or validation data.
- **License compliance** (`use.txt` in the zip = Microsoft Research License): non-commercial research use only; the data and any modification of it (videos, frames, extracted keypoints, keypoint drawings) are never distributed, so they stay under `data/public/` (gitignored) and out of `docs/`, reports and commits; personal data is not shared or used to contact anyone; a publication may not contain a substantial portion of the data; the original zip stays outside the repo (`C:\dev\Datasets`) and the extracted personal data is destroyed when the research is finished.

- `client/recorder.py` (or `client.py --record <label> --signer <id>`): shows a countdown, then records a **3-second** clip at the live FPS and resolution, using the **same JPEG encoder and quality as live streaming**.
- Each clip is a folder: `data/raw/<label>/<signer>_<YYYYmmdd-HHMMSS>/000.jpg ... 044.jpg` plus `meta.json` (`label, signer, fps, width, height, jpeg_quality, created`).
- Recording routine shown on screen: hands down, then sign, then hands down.
- Add a quick review mode that replays a clip and lets the user delete bad ones.

## 7. C++/CUDA requirements

**Build**
- CMake presets for Ninja and the MSVC toolchain, `CMAKE_CUDA_ARCHITECTURES=120`, C++20, CUDA flags `-O3 -lineinfo`. Use `--use_fast_math` only on kernels whose tests pass with it, and record that in `DECISIONS.md`.
- Dependencies through a **vcpkg manifest**: a WebSocket/HTTP server library (prefer ixwebsocket; Boost.Beast if needed, and record the reason), nlohmann-json, spdlog, GoogleTest. CUDA, nvJPEG and TensorRT are found from their install folders.
- At configure time, verify the TensorRT version and that headers and libs are found. Fail with a clear message otherwise.
- Post-build: copy the required DLLs next to the exe so it runs from a plain terminal.

**Startup (in order, fail fast with clear logs)**
1. Check the GPU and compute capability (needs sm_120 or compatible).
2. `cudaMemGetInfo`: refuse to start if free VRAM is below the budget (default: need at least 4 GB free, cap own usage at 10 GB; configurable in `server.json`).
3. Load engines. If an engine is missing or was built for a different TensorRT version or GPU, log the exact `build_engines.ps1` command to fix it.
4. Preallocate every buffer: pinned host staging, device frame buffers for max resolution, model I/O, ring buffer, nvJPEG state. **Zero allocations per frame** in steady state (assert this in debug builds with a counter).
5. Start the WebSocket server on `127.0.0.1:8765` (configurable). It binds to localhost only, because the tunnel connects locally.

**Hot path performance rules**
- No `cudaDeviceSynchronize`, `cudaMalloc`/`cudaFree`, or synchronous memcpy per frame. Use streams with events, `cudaMemcpyAsync` from pinned memory, and one tiny device-to-host copy at the end.
- **Fused preprocess kernel:** crop/affine plus bilinear resize, BGR/RGB conversion, normalization, and HWC uint8 to the model's input layout and dtype, in one pass that writes directly into the TensorRT input binding. Replicate the keypoint model's **exact** preprocessing (affine transform, padding or scale factor, mean/std, channel order, pixel-center convention), read from the model's own config, not guessed. Document the source in `DECISIONS.md`.
- Coalesced, vectorized loads where it helps, `__restrict__`, and boundary handling for ROIs partly or fully outside the frame.
- Box selection, keypoint decoding (for example SimCC argmax if the model uses it), keypoint normalization (center on mid-shoulder, scale by shoulder width, plus a presence mask for missing hands), ring-buffer write, and softmax/EMA all run on the GPU. ROI parameters never round-trip through the CPU.
- After Phase 6 works, capture the steady-state pipeline in a **CUDA Graph** and keep the non-graph path behind a flag.
- NVTX ranges on every stage. CUDA-event timings per stage go in the reply's `server_ms`. `--bench N` mode prints p50/p95/p99 per stage.

**Robustness**
- A `CUDA_CHECK` / `TRT_CHECK` macro on every API call, with file, line and error string.
- Catch allocation and TensorRT failures at startup and runtime. Runtime degrade (tier 1): switch to a lower resolution and the small fallback engine if latency exceeds the budget for N frames, VRAM gets low, or an allocation fails. Never crash on a bad frame.
- `--mode full|safe`: safe mode uses the small engine and a lower resolution.
- Graceful shutdown on Ctrl+C. No leaks (verify with compute-sanitizer and a 10-minute soak test).

**Performance targets** (internal gates, report measured numbers in `STATUS.md`; no comparison with Python)

| Stage | Target |
|---|---|
| nvJPEG decode 720p | about 3 ms or less |
| Fused preprocess | under 0.5 ms |
| Keypoint stage (all models) | about 12 ms or less |
| Temporal model | about 1 ms or less |
| Post-process + smoothing | under 0.1 ms |
| Total GPU time per frame | about 20 ms or less |
| End to end, laptop to laptop, real network | under 100 ms |

If a target is missed, profile with Nsight Systems or Compute, fix the top bottleneck, and record before and after numbers. Don't just report the miss.

## 8. Decision logic (CPU, tiny, unit-tested; reference implementation `server/fallback/decision.py`)

- Sliding window of 30 frames = 3 s at 10 fps (configurable; set by the clip data, see Phase 7), run on every new frame. Frames are placed on a 100 ms grid from the client timestamps, so dropped frames do not stretch time.
- **Idle gate (rule based):** a hand must be strictly visible in at least `gate_min` of the last `gate_frames` frames (defaults 2 of 5), otherwise the state is `idle`, nothing commits and the smoothing resets.
- Smooth the probabilities (EMA, alpha 0.5-0.7). Commit a word only after `M` consecutive frames whose smoothed best non-`none` class agrees with confidence of at least `threshold` (M, threshold and alpha are tuned on the validation signers; current defaults M=4, threshold 0.5, alpha 0.7).
- **Cooldown** after a commit (default 800 ms) and **re-fire protection**: a committed word stays "fired" until another word has been predicted for 3 frames, `none` for 8 frames, or the gate closes, so one sign never fires twice.
- **Word joining:** `how` then `you` = `how_are_you`; `where` then `you`, or `you` then `where` = `where_are_you`. `how`, `where` and `you` are held up to `join_ms` (default 1500) for a partner, measured from the moment the first word stops being predicted, then emitted alone. At most one event per frame.
- All parameters live in `server.json` (per profile overrides allowed). Tests feed synthetic probability sequences and check the committed outputs, including the join cases, a held sign (no double fire), the cooldown and the idle gate.

## 9. Training (Python, offline, GPU-resident)

- `extract_keypoints.py`: turns recorded clips into keypoint sequences using the **same ONNX keypoint models and identical preprocessing** as runtime (onnxruntime-gpu). Later, the C++ exe gets an `--extract <clips_dir> <out_dir>` mode that writes the same format, and the final model is trained on C++-extracted keypoints so training matches runtime exactly.
- `train.py`: load all sequences into VRAM. Augment on the GPU: random time shift, speed 0.8 to 1.2x via resampling, frame drop, keypoint jitter, small scale and rotation. Mirroring stays off by default, because it changes handedness. Use AMP (bf16), a GRU or small Transformer (record the choice), class weights for imbalance, and early stopping.
- **Data:** ASL Citizen only (section 6), using its official split, which is already signer-disjoint (train 35 / val 6 / test 11 signers).
- **Validation is held out by signer** (one signer never seen in training). Print and save per-class accuracy and a confusion matrix.
- `export_onnx.py`: fixed input shape `[1, 45, F]`, opset compatible with the installed TensorRT, verify ONNX vs PyTorch outputs match within tolerance.
- `make_placeholder_model.py`: creates a correctly shaped temporal ONNX model with random weights, so the full pipeline runs before any data exists. Mark it clearly as a placeholder everywhere it is used.
- `training/requirements.txt` pins versions that actually installed and worked on this PC.
- Smoke test: the whole training pipeline runs end to end on **synthetic** sequences before real data exists.

## 10. Python fallback server (minimal)

- `server/fallback/fallback_server.py`: FastAPI plus uvicorn, the same `/ws`, `/health`, auth and replies as the C++ server, using onnxruntime-gpu with the CUDA execution provider and the same ONNX models, preprocessing, decision logic and `server.json`. Reuse code from `training/` where possible. Target about 300 lines or fewer.
- Verify `CUDAExecutionProvider` is actually used, not silently falling back to CPU. Log the active provider at startup.
- It must pass `conformance_test.py`. It is never benchmarked against the C++ server.

## 11. Laptop client (Python, CPU only)

- `client.py --url wss://<TUNNEL_HOSTNAME>/ws` (token from `SIGNLANG_TOKEN`): capture, resize, JPEG encode, send at the target FPS in a separate thread from receiving, so a slow network never freezes the preview.
- Overlay: current label, confidence bar, committed label history, round-trip latency, server-side timings, FPS, and connection state. Auto-reconnect with backoff.
- `config.json`: URL, resolution, FPS, JPEG quality, camera index. Packages are `opencv-python`, `websockets` and `numpy` only.
- Must run on a CPU-only laptop. Test it on this PC with the GPU unused by the client process.

## 12. Tests and quality gates

- **Kernels (GoogleTest):** a CPU reference for each kernel. Tolerance: at most 1 LSB for uint8 outputs, about 1e-3 for FP16, documented per test. Cases: ROI fully inside, touching the border, crossing the border, fully outside, zero-size ROI, odd sizes, sizes not divisible by the block size, max resolution, NaN/Inf in float inputs. Add randomized tests with a fixed seed.
- **compute-sanitizer** (memcheck, racecheck, initcheck, synccheck) on the kernel tests and a short pipeline run, via `scripts/sanitize.ps1`. It must be clean.
- **Parity:** `parity_check.py` runs the same recorded clips through the Python/ORT extractor and the C++ `--extract` mode. Keypoints must match within a documented tolerance, and the final labels must match.
- **End to end:** replay recorded clips through a headless client against each server, and check the committed labels.
- **Soak:** 10 minutes of streaming. VRAM stays flat, there are no errors, and no latency creep.
- **Failure injection:** missing engine, forced allocation failure, oversized and garbage messages, client disconnect mid-frame, tunnel drop. The server must log, degrade or reject cleanly, and keep running.
- `scripts/run_tests.ps1` runs everything automated in one command.

---

## 13. Phases and gates

**Order changed 2026-11-18 (deadline Nov 28): Phases 7 and 8 come BEFORE Phases 4-6.** The real model and decision logic go into the
Python fallback server first, so there is a working demo early. The C++/CUDA server (Phases 4, 5, 6) follows and reuses the same models,
keypoint features and decision logic (`server/fallback/decision.py` is the reference for the C++ port; parity tests in Phase 6/8).
Phase 3 (done, approved): primary RTMW-x(l) 384x288 on the full frame, TensorRT FP16; fallback RTMW-l-m 256x192; YOLOX-tiny optional and off by default.
Trained weights stay local-only (dataset license). Real order now: 0, 1, 2, 3, 7, 8, 4, 5, 6, 9, 10.

Do them in order. At each gate: run the listed checks, paste the real output summary, update `STATUS.md`, commit, then continue. If a gate fails, fix it before moving on.

**Phase 0: Preflight and skeleton**
Verify the toolchain (`nvidia-smi`, `nvcc --version`, `cl`, `cmake --version`, `python --version`, the torch CUDA check) and free disk space. Create the repo at `C:\dev\almighty-sign-language`, the layout from section 4, `.gitignore`, `CLAUDE.md`, `STATUS.md`, `DECISIONS.md`, `protocol/labels.json`, `protocol/PROTOCOL.md`.
*Gate:* all tool checks printed. Repo skeleton committed.

**Phase 1: Dependencies**
Install vcpkg (in `C:\dev\vcpkg`) and the manifest packages. **Stop and ask the user** to download the TensorRT Windows zip that supports CUDA 13.x and Blackwell (sm_120) from NVIDIA's developer site, and to tell you where they unzipped it. Verify against the release notes. Set up Python venvs (`training/.venv` for training and fallback). Install pinned packages. Verify torch CUDA, and that onnxruntime-gpu really runs on `CUDAExecutionProvider` (if its CUDA/cuDNN DLLs are missing, resolve it properly and document how).
*Gate:* a tiny C++ program built with CMake links CUDA, nvJPEG and TensorRT and prints all versions and the GPU name. `trtexec` builds an FP16 engine from a tiny ONNX model and runs it.

**Phase 2: Protocol, client and fallback skeleton**
Client with live view, overlay and record mode. Fallback server using the placeholder temporal model and a dummy keypoint step. `conformance_test.py`.
*Gate:* conformance passes against the fallback locally. The client round trip works locally. **Ask the user for the tunnel hostname**, then verify the client works through `wss://<hostname>/ws`.

**Phase 3: Keypoint model selection (decision gate)**
Research and test candidates that output **both hands (21 keypoints each) plus upper body**. Requirements: ONNX export or download, builds as a TensorRT FP16 engine on sm_120, a license that allows academic use, and the fewest models in the stage. Evaluate RTMPose/RTMW-style whole-body models with a person detector first. A MediaPipe-derived model is a backup only, and never runs as MediaPipe at runtime. For each candidate: run ORT on dataset clips converted to the live format (section 6) and on the live webcam preview (view only, nothing saved), draw the keypoints, build the TRT engine, and compare TRT against ORT outputs.
*Gate:* write the comparison (accuracy on webcam frames, latency, model count, export problems) in `DECISIONS.md`. **Stop and get the user's approval** before continuing.

**Phase 4: C++ server core**
WebSocket server, auth, health, latest-frame slot, config loading, startup checks and preallocation, nvJPEG decode, logging, error replies. Returns dummy results.
*Gate:* both the C++ server and the fallback pass `conformance_test.py`. The client gets results from the C++ server through the tunnel.

**Phase 5: Kernels**
Fused preprocess, detector post-process, keypoint decode and normalize, ring buffer, softmax/EMA. CPU references and tests first, then the kernels.
*Gate:* all kernel tests pass, sanitizer clean, Nsight Compute numbers for the preprocess kernel recorded.

**Phase 6: Full C++ pipeline (placeholder temporal model)**
TensorRT engine wrapper (load, bind, enqueueV3 on streams), all stages wired, per-stage timings, `--bench`, `--extract` mode.
*Gate:* live end to end through the tunnel with real keypoints drawn on the client (send keypoints back in a debug field behind a flag). `--bench` numbers recorded. `parity_check.py` passes on at least 5 clips.

**Phase 7: Training pipeline**
Extract, dataset, train, export and engine build, on synthetic data first.
*Gate:* the synthetic smoke test trains, exports, builds the engine, and loads in both servers. Then train on the real public data (ASL Citizen, extracted keypoints). Nobody records data (section 6, data policy).

**Phase 8: Decision logic**
State machine with prefix hold and cooldown, plus unit tests.
*Gate:* decision tests pass in C++ and Python, and both implementations produce the same commits for the same synthetic sequences.

**Phase 9: Robustness and performance**
Tier 1 degrade and the small engine, `--mode safe`, CUDA Graph, failure-injection tests, soak test, profiling pass against the targets.
*Gate:* all failure-injection cases handled, soak clean, measured numbers in `STATUS.md`.

**Phase 10: Scripts and docs**
`start_server.ps1`, `start_fallback.ps1`, `build_engines.ps1`, `run_tests.ps1`, `sanitize.ps1`. A README with step-by-step setup for the PC and for the laptop. A demo-day checklist (start order, how to switch to the fallback, how to rebuild engines after a driver update).
*Gate:* a fresh clone on this PC builds and runs by following the README alone.

**Optional, only if everything above is done:** tier 2 (ONNX Runtime C++ instead of TensorRT) and tier 3 (unfused kernels) fallbacks, H.264/NVDEC.

---

## 14. When the real model is trained (after data exists)

Retrain on C++-extracted keypoints, export, build engines, run parity and the end-to-end replay tests, report the held-out-signer accuracy and confusion matrix, and tune `threshold`, `M`, `hold_ms` and `cooldown` in `server.json` from that confusion matrix. Report the final numbers honestly, including weak classes.
