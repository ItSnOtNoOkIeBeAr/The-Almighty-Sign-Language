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

## Phase 2: Protocol, client, fallback skeleton: DONE (2026-10-21)

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

Tunnel gates (hostname `signlang.dheynorlanda.me`, Windows service `cloudflared` RUNNING, route -> `http://localhost:8765`):

| Gate item | Command | Result |
|---|---|---|
| Health through tunnel | `GET https://signlang.dheynorlanda.me/health` | HTTP 200 `{"ok":true,"server":"fallback","version":"0.1.0"}`; DNS resolves to Cloudflare edge IPs |
| Conformance through tunnel | `conformance_test.py --url wss://signlang.dheynorlanda.me/ws` | **9/9 PASS**; rapid fire 60 sent / 38 processed / 22 dropped. Authorised tests passing proves the `Authorization` header survives the tunnel; 401 for missing/wrong token also arrives intact |
| Client through tunnel | `client.py --url wss://signlang.dheynorlanda.me/ws --headless --duration 12 --width 640 --height 480` | 172 frames sent, 172 results, **RTT p50 55.2 ms / p95 61.8 ms / max 103.8 ms**, server total p50 3.5 ms |
| Localhost regression after test change | conformance on `ws://127.0.0.1:8765/ws` | 9/9 PASS |

First tunnel run failed 1 of 9: `/health` returned 403. Cause (verified): Cloudflare error 1010 blocks urllib's default `Python-urllib` User-Agent at the edge (server header `cloudflare`, request never reached the app); an explicit User-Agent returns 200. Fixed in the test, nothing changed on the Cloudflare side. The first `service install` attempts failed with a truncated token and then a non-admin shell; the user ran it from an admin shell.

**Not verified:**
- Interactive live window (overlay drawing): no human at the screen; only headless mode was run. Review-mode key handling was tested with simulated keys, not a real keyboard.
- Client on a *separate* CPU-only laptop not yet run (the laptop needs `SIGNLANG_TOKEN` set and `pip install -r client/requirements.txt`).
- The RTT above is PC -> Cloudflare edge -> PC, not laptop-to-laptop over a real network (that target, under 100 ms end to end, is measured in Phases 6/9).
- Cloudflare dashboard flags the installed cloudflared (2025.8.1) as "Unsupported" (old). It works; update it if the tunnel misbehaves.
Notes:
- Webcam on this PC is the "A4tech PC Camera" at index 0, 640x480 max, and dim: ~36 distinct frames per 45 slots (~12 fps). Index 1 is the OBS Virtual Camera (not used).
- Pacing bug found and fixed during testing: the first recorder waited for each camera frame, so a slow camera stretched clips to 4.2 s while `meta.json` claimed 15 fps. Now a capture thread supplies the newest frame at each 1/15 s tick, like live streaming; measured capture span 2.93 s for 45 frames.
- uvicorn logs a false ERROR after every correct 401 denial; filtered (see DECISIONS).

## Phase 2 follow-up: laptop over real Wi-Fi (2026-10-27)

Laptop (Killer Wi-Fi 6E AX1675i, Windows-reported link 14.4 Mbps) through the tunnel, 20 s headless runs, flags only:

| Settings | Frame size | Sent / results | RTT p50 / p95 / max (ms) | Trend (first 5 -> last 5 rtt) |
|---|---|---|---|---|
| 1280x720, q80, 15 fps (config defaults) | 80.7 KB | 199 / 172 | 1421 / 4531 / 5765 | 403-848 -> 2812-2891 (**grows**) |
| 640x360, q70, 15 fps | 22.4 KB | 200 / 175 | 455 / 623 / 738 | flat, ~500-700 |
| 640x360, q70, 8 fps | 22.4 KB | 158 / 157 | 142 / 249 / 445 | flat, ~120-200 |

Cause: the stream (80.7 KB x 15 = 9.7 Mbit/s) exceeds what the Wi-Fi uplink sustains, so frames queue in network buffers and RTT grows without bound. Server time stayed ~4-8 ms throughout, so the server is not involved. A first 8 fps attempt also hit a DNS error (`gaierror`): the laptop's network drops out now and then (reconnect logic handles it).

Fix (client): flow control, `max_inflight` (default 3, 0 = off) in `client/config.json` / `--max-inflight`: skip sending while more than N frames are unanswered, with a 2 s escape hatch. Verified with a throttling proxy on this PC (60 KB/s, unbounded buffer, scratchpad `slowlink.py`, not in the repo):

| Link | max_inflight | RTT first 5 -> last 5 (ms) | p50 / p95 / max (ms) | Skipped |
|---|---|---|---|---|
| 60 KB/s | 0 (old) | 115 -> 8637 | 4363 / 8419 / 8806 | 0 |
| 60 KB/s | 3 (new) | 130 -> 243 | 330 / 416 / 434 | 130 |
| localhost | 3 | flat | 5.6 / 8.5 / 17.0 | 0 |
| this PC via tunnel | 3 | flat | 102.7 / 110.4 / 120.0 | 0 |

Not yet verified on the real laptop: the flow-control build has not been run there (needs `git pull` on the laptop).
OPEN DECISION (user): final live stream settings (resolution / JPEG quality / fps). Clips must be recorded with the SAME settings
(spec section 6), and the fps also sets how many frames a 3 s clip has versus the 45-frame window. Record NO real training clips until decided.
## Placeholders
- `training/make_placeholder_model.py` / `server/models/temporal_placeholder.onnx`: random-weight GRU, recognises nothing (replaced in Phase 7).
- `server/fallback/fallback_server.py`: dummy keypoint step (8x12 grayscale thumbnail = 96 features) and a decision step that never commits (real logic in Phase 8). Marked `PLACEHOLDER` in code.

## Known issues
- `cl`/`cmake` are only available after `vcvars64.bat`; handled by scripts.
