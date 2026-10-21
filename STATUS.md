# STATUS

## Phase 0: Preflight and skeleton: DONE (2026-10-21)

Verified (in a plain PowerShell terminal unless noted):

| Check | Command | Result |
|---|---|---|
| GPU / driver | `nvidia-smi` | RTX 5060 Ti, driver 617.42, CUDA UMD 13.4, 16311 MiB |
| CUDA toolkit | `nvcc --version` | 13.2, V13.2.51 |
| Host compiler | `vcvars64.bat && cl` | MSVC 19.42.34436 x64 (not on default PATH) |
| CMake | `vcvars64.bat && cmake --version` | 3.29.5-msvc4 (not on default PATH) |
| Ninja | `vcvars64.bat && where ninja` | VS-bundled ninja.exe |
| Python | `python --version` | 3.11.9 |
| Torch CUDA | `python -c "import torch; ..."` | 2.11.0+cu128, `cuda.is_available() == True`, RTX 5060 Ti |
| Git | `git --version` | 2.47.1 |
| Disk | `Get-PSDrive C` | 174.6 GB free |

Created: section 4 layout, `.gitignore`, `CLAUDE.md`, `STATUS.md`, `DECISIONS.md`, `README.md`,
`protocol/labels.json`, `protocol/PROTOCOL.md`.

Not verified: n/a.

## Phase 1: Dependencies: DONE (2026-10-21)

| Check | Command | Result |
|---|---|---|
| vcpkg | `vcpkg install --triplet x64-windows` (manifest `server/cpp/vcpkg.json`) | ixwebsocket 12.0.1 (no SSL), nlohmann-json, spdlog 1.17.0, gtest installed |
| TensorRT | `NvInferVersion.h`, `trtexec` | 10.16.1.11 at `C:\dev\TensorRT-10.16.1.11` (zip for CUDA 13.0-13.2); zip ships `nvinfer_builder_resource_sm120_10.dll` |
| FP16 engine | `trtexec --onnx=tiny.onnx --fp16 --saveEngine=tiny.engine` | Built in 31 s, runs, GPU compute mean 0.058 ms |
| ORT on GPU | `training/ort_cuda.py` `make_session()` | `['CUDAExecutionProvider','CPUExecutionProvider']`; raises if it would fall back to CPU |
| Link test | CMake+Ninja+MSVC, `server/cpp/tools/link_check` | RTX 5060 Ti sm_120, CUDA driver 13.4, nvJPEG 13.0.4, TensorRT runtime 101601; exit 0 |
| Python env | `training/requirements.txt` | torch 2.11.0+cu128 (CUDA True), onnxruntime-gpu 1.31.0 etc. pinned |

Not verified / notes:
- **CUDA version mismatch vs expectation (unresolved):** the updated link test prints driver and runtime separately: `cudaDriverGetVersion` = 13040 (13.4, as expected) but `cudaRuntimeGetVersion` = 13040 (13.4), NOT the expected 13.2. Facts gathered: only toolkit v13.2 is installed, `CUDART_VERSION` in its headers = 13020, `version.json` says 13.2.0 / cudart 13.2.51, nvcc is 13.2.51, and the exe links cudart statically (no cudart DLL dependency per `dumpbin`). A bare nvcc program prints the same 13040. Cause not found; no functional problem seen (kernels/TensorRT/nvJPEG all run). Will recheck if a kernel/runtime mismatch ever appears.
- Engines are tied to this TensorRT version and GPU; rebuild after upgrading either.
- `training/.venv` is the shared venv for training and the fallback server. The laptop client will get its own CPU-only requirements in Phase 2.

## Phase 2: Protocol, client, fallback skeleton: PARTIAL (tunnel gates pending)

| Gate item | Command | Result |
|---|---|---|
| Port free | `netstat -ano \| findstr :8765` | free before start; both servers use `127.0.0.1:8765` (spec, PROTOCOL.md, `server/config/server.json` updated) |
| Fallback uses GPU | server startup log | `onnxruntime active providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']` |
| Conformance, fallback, localhost | `client\.venv\Scripts\python protocol\conformance_test.py --url ws://127.0.0.1:8765/ws` | **9/9 PASS** (health, 401 missing/wrong token, round trip, 6 malformed cases, oversized 2 MB+1 and 5 MB, EOS, rapid fire 60 sent / 17 processed / 43 dropped, reconnect incl. abrupt drop) |
| Client round trip, localhost | `client\.venv\Scripts\python client\client.py --headless --duration 8 --width 640 --height 480` | 112 frames sent, 112 results, RTT p50 5.5 ms / p95 6.0 ms / max 12.0 ms, server total p50 3.6 ms. Labels all `idle` (placeholder model) |
| Client venv is CPU-only | `pip list` in `client/.venv` | numpy 2.4.6, opencv-python 5.0.0.93, websockets 17.3 only; torch/onnx/onnxruntime/tensorrt NOT importable |
| Record mode | `client.py --record <label> --signer test1 --headless --width 640 --height 480` | 4 test clips (hello, hi, why x2; signer `test1`): 45 JPEGs `000..044.jpg` + `meta.json` each, 640x480, 45 slots at 15 fps = 2.93 s capture span (measured), git-ignored |
| Review/delete mode | real `recorder.review()` with simulated keys on a scratchpad copy | `n` next, `d`+`x` cancel (nothing deleted), `d`+`y` deleted the clip; real clips untouched |
| SIGNLANG_TOKEN | `setx`, read back from HKCU | 43 URL-safe chars, stored as a user env var; never written to the repo |

**NOT verified (blocked):**
- Conformance and client **through the Cloudflare tunnel**: `cloudflared.exe` exists (`C:\Program Files (x86)\cloudflared\`) but no `cloudflared` Windows service is registered (`sc query cloudflared` -> error 1060), no process is running, and the public hostname was never provided (the instruction text still said `<YOUR_HOSTNAME>`). Nothing on the Cloudflare side was touched.
- Interactive live window (overlay drawing) not exercised: no human at the screen; only headless mode was run. Review mode key handling was tested with simulated keys, not a real keyboard.
- End-to-end client on a *separate* CPU-only laptop not yet run.

Notes:
- Webcam on this PC is the "A4tech PC Camera" at index 0, 640x480 max, and dim: ~36 distinct frames per 45 slots (~12 fps). Index 1 is the OBS Virtual Camera (not used).
- Pacing bug found and fixed during testing: the first recorder waited for each camera frame, so a slow camera stretched clips to 4.2 s while `meta.json` claimed 15 fps. Now a capture thread supplies the newest frame at each 1/15 s tick, like live streaming; measured capture span 2.93 s for 45 frames.
- uvicorn logs a false ERROR after every correct 401 denial; filtered (see DECISIONS).

## Placeholders
- `training/make_placeholder_model.py` / `server/models/temporal_placeholder.onnx`: random-weight GRU, recognises nothing (replaced in Phase 7).
- `server/fallback/fallback_server.py`: dummy keypoint step (8x12 grayscale thumbnail = 96 features) and a decision step that never commits (real logic in Phase 8). Marked `PLACEHOLDER` in code.

## Known issues
- `cl`/`cmake` are only available after `vcvars64.bat`; handled by scripts.
