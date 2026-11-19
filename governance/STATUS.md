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
DECIDED (Claude, user delegated): live stream = 640x360, JPEG q70, 10 fps, max_inflight 3 (~1.8 Mbit/s). PENDING laptop confirmation that RTT stays flat; fall back to 8 fps if p50 > 200 ms. Clips use the SAME settings (spec section 6): 3 s = 30 frames at 10 fps (the server's 45-frame window is revisited in Phase 7). Record real training clips only after the laptop confirms.
## Placeholders
- `training/make_placeholder_model.py` / `server/models/temporal_placeholder.onnx`: random-weight GRU, recognises nothing (replaced in Phase 7).
- `server/fallback/fallback_server.py`: dummy keypoint step (8x12 grayscale thumbnail = 96 features) and a decision step that never commits (real logic in Phase 8). Marked `PLACEHOLDER` in code.

## Known issues
- `cl`/`cmake` are only available after `vcvars64.bat`; handled by scripts.

## Laptop check and first clips (2026-10-27)

- Laptop client venv: Python 3.12.10, only numpy, opencv-python, websockets (no torch/onnxruntime/tensorrt). `SIGNLANG_TOKEN` set, length 43. Camera index 0: 640x480 (4:3), ~29.8 fps measured over 3 s.
- Config pulled (`71deca6`); `/health` 200 through the tunnel. `run.ps1` fixed (config-driven, passes extra args).
- 4:3 squash found: 640x360 stretched the 640x480 camera. Config changed to 640x480 (see DECISIONS).
- 20 s headless runs over the tunnel with flow control: 640x360 10 fps p50 118 ms (flat); 640x360 8 fps p50 132 ms; 640x480 10 fps p50 447 / 136 / 132 / 465 ms (bimodal, never growing; skips 76 / 3 / 2 / 77). Wi-Fi variance, not frame size. Live overlay confirmed working by the user.
- Server labels were not always `idle` (random placeholder model returned hello/how/hi).
- Recorded 5 sample clips (signer `me`, 640x480, q70, 10 fps): hello, how, how_are_you, where_are_you, idle. Each 30 JPEGs, 30 distinct camera frames, ~28-32 KB per JPEG, no low-light warning. Stored in `data\raw` (gitignored) on the laptop; not in the repo.
- Not yet verified: a stable p50 <= 200 ms on the laptop Wi-Fi; the clips have not been reviewed (`--review`).

## Step A: ASL, public data only (2026-11-18)
- Done: BUILD_SPEC sections 1, 6, 8, 9 and Phases 3/7 updated (ASL, public data only, word-joined sentences, idle/other, domain matching, live-check mode planned, license compliance); DECISIONS entry; the 5 laptop clips deleted from `data/raw` (git will show them as deleted: they had been committed; 4 `test1_*` clips remain).
- Not done: `protocol/labels.json` still the original 8 classes (see DECISIONS); `client.py --live-check` not written (PLACEHOLDER: spec only).

## ASL Citizen data (2026-11-18)
- Zip `C:\dev\Datasets\ASL_Citizen.zip` (45.9 GB): a full CRC test of all members passed (`zipfile.testzip()` found no bad member, 224 s). It holds 83,399 mp4, 3 split CSVs, `use.txt` (the README: Microsoft Research License, see DECISIONS Step A) and 3 folders. CSV header `Participant ID,Video file,Gloss,ASL-LEX Code`; 52 signers (train 35 / val 6 / test 11, no overlap), 2,731 glosses.
- Selective extraction (`training/extract_asl_citizen.py`, seed 1234) into `data/public/asl_citizen/` (gitignored, checked with `git check-ignore`): 4,826 mp4, 2.86 GB. Each file's CRC32 was verified on read and against the zip directory; `manifest.csv` keeps the official split.

| Gloss (class) | clips | signers | train / val / test | resolution | fps | frames median (min-max) |
|---|---|---|---|---|---|---|
| HELLO (hello) | 32 | 29 | 14 / 4 / 14 | 640x480 (31), 960x540 (1) | 30 / 29.97 | 65 (38-170) |
| HOW1 + HOW2 (how) | 31 + 31 = 62 | 29 / 27 | 17+14 / 3+3 / 11+14 | 640x480 (59), 960x540 (3) | 30 / 29.97 / 29.92 | 72 / 57 |
| WHAT1 + WHAT2 (what) | 32 + 31 = 63 | 27 / 26 | 16+15 / 3+4 / 13+12 | 640x480 (61), 960x540 (2) | 30 / 29.92 | 64 / 70 |
| WHY (why) | 32 | 29 | 14 / 4 / 14 | 640x480 (31), 960x540 (1) | 30 / 29.92 | 68 (28-126) |
| WHERE (where) | 31 | 29 | 14 / 4 / 13 | 640x480 (30), 960x540 (1) | 30 / 29.97 | 61 (24-158) |
| YOU (you) | 31 | 30 | 15 / 4 / 12 | 640x480 (30), 960x540 (1) | 30 / 29.92 | 68 (21-160) |
| OTHER: 150 random glosses (idle/other) | 4,575 | 50 | 2204 / 563 / 1808 | 640x480 (~96%), 960x540 | 30 / 29.97 | 75 (27-291) |

- Every target has at least 31 clips from at least 26 signers (the bar was 20 clips / 8 signers), so Phase 3 continued without waiting. There is **no separate HI gloss** (nearest: HELLO, WAVE, GREET1/2). `how` and `what` each have two gloss variants (HOW1/2, WHAT1/2), both extracted into one class. **YOU is labelled `YOU`** (YOUR, YOURSELF, SHAMEONYOU, THANKYOU, ILOVEYOU and TENDYOUR also exist). OTHER excludes any gloss containing HELLO/HOW/WHAT/WHY/WHERE/YOU/WAVE/GREET. The val split has only ~3-4 clips per word, so word-level val numbers will be noisy (use val+test or signer-grouped cross-validation).
- Clips are ~2 s of 30 fps video (median ~65 frames), i.e. ~22 frames at the 10 fps live rate; the 45-frame window in `server.json` is a placeholder to revisit in Phase 7.

## Phase 3: keypoint model selection: DONE, AWAITING USER APPROVAL (2026-11-18)
Full comparison, exact preprocessing, export problems and the recommendation are in DECISIONS "Phase 3". Verified:

| Check | Command / file | Result |
|---|---|---|
| 5 pose + 2 detector candidates downloaded | `server/models/candidates/` (2.18 GB, gitignored) | all HTTP 200 |
| ORT on GPU, 50 clips / 1399 frames, 16 detector x pose combos | `python training/phase3_eval.py ort` | tables in DECISIONS |
| TRT FP16 engines on sm_120 | `trtexec --fp16`, 68-117 s each | 7/7 built (YOLOX-m only after the TopK patch) |
| TRT vs ORT keypoints (64 crops) and boxes (16 frames) | `phase3_eval.py trtcmp`, `server/scripts/phase3_trt_compare.ps1` | mean ~0.4 px, p95 1.2-2.2 px, 0.16-3.6% of keypoints >5 px; boxes <= 0.65 px |
| Drawings | `phase3_eval.py draw` | `data/public/asl_citizen/phase3/drawings/*.jpg` |

Not verified: hand-detection accuracy against ground truth (no labels exist; proxies only); the live webcam check (image almost black, see DECISIONS); TensorRT inside the future C++ pipeline (Phases 5/6); end-to-end sign accuracy (Phase 7). `protocol/labels.json` is still the original 8 classes.
New code: `training/{extract_asl_citizen,rtmw_ref,phase3_eval,phase3_live_view}.py`, `server/scripts/phase3_trt_compare.ps1`. PLACEHOLDER: none added.

## Step 1: privacy check of recorded clips (2026-11-19)
- `git log --all -- data/raw`: two commits, both on `main` and **both pushed** (`origin/main` = local `main` at `3452a31` when checked): `8b31393` (155 files: the 5 laptop clips `me_20261027-*`, which show the room and other people) and `3452a31` (184 files: the 4 `test1_*` clips). Cause: `.gitignore` had `!data/raw/`.
- Done locally: all 9 clips deleted (339 files, shown as deletions in `git status`); `data/raw/` is now gitignored (`git check-ignore -v` confirms).
- NOT done (user decides): history purge. Proposed commands (run from a fresh clone; `pip install git-filter-repo`):
  ```
  git clone --mirror https://github.com/ItSnOtNoOkIeBeAr/The-Almighty-Sign-Language.git backup.git      # safety copy first
  git clone https://github.com/ItSnOtNoOkIeBeAr/The-Almighty-Sign-Language.git purge && cd purge
  git filter-repo --invert-paths --path data/raw --path LAPTOP_SETUP.md      # LAPTOP_SETUP.md also held the real SIGNLANG_TOKEN
  git remote add origin https://github.com/ItSnOtNoOkIeBeAr/The-Almighty-Sign-Language.git             # filter-repo removes the remote
  git push origin --force --all
  ```
  Then everyone re-clones (old clones still hold the clips). GitHub may keep the old commits reachable by SHA and in forks until it garbage-collects (contact GitHub support to purge cached views). **Making the repo private also hides them from the public; do that now.** Rotating `SIGNLANG_TOKEN` is still advisable.

## Phase 7: training: DONE (2026-11-19) and Phase 8: decision logic: DONE (2026-11-19)
Details, numbers and every non-obvious choice: DECISIONS "Phase 7/8". Verified:

| Check | Command | Result |
|---|---|---|
| Keypoints, both models | `python training/extract_keypoints.py` | 4,826 clips, 138,521 frames; shards verified (rows == index) |
| Training, 8 seeds x 2 keypoint models | `python training/train.py --model <m> --seeds 8` | 1,315,015 params, ~1 min/seed |
| **TEST signers, window level, deployed primary (RTMW-x 384x288)** | `python training/report_test.py --model rtmw_xl_384x288` | accuracy 0.848, **macro-F1 0.828** (533 windows, 11 signers); per-class recall none .951, hello .929, what .800, how .813, why .762, where .692, you .556 |
| TEST, deployed fallback (RTMW-l-m 256x192) | same, `--model rtmw_lm_256x192` | accuracy 0.852, macro-F1 0.816; recall none .960, hello .786, what .880, how .893, why .595, where .615, you .667 |
| End-to-end TEST replay (clips as 10 fps streams through features + model + decision), primary | `python training/replay_eval.py --model rtmw_xl_384x288 --tune` | words exactly right 73/103 (70.9%): hello 13/14, how 15/25, what 18/25, why 10/14, where 9/13, you 8/12; false triggers on 400 non-target glosses 15 (3.7%); sentences 8/33 (24%): how_are_you 3/11, where_are_you 5/22 |
| same, fallback | `--model rtmw_lm_256x192` | words 70/103 (68.0%), false triggers 26/400 (6.5%), sentences 5/33 (15%) |
| ONNX vs PyTorch | `python training/export_onnx.py --model <m> --out server/models/temporal_<profile>` | max logit diff 9e-6 (exact FP32), argmax 100%; TensorRT FP16 engine built, diff 0.004 / 0.011, GPU 0.475 ms |
| Decision unit tests | `python -m unittest server/fallback/test_decision.py -v` (training venv) | 20/20 pass |
| Conformance, real model, localhost | `client\.venv\Scripts\python protocol\conformance_test.py --url ws://127.0.0.1:8765/ws` | **9/9 PASS** (rapid fire 60 sent / 6 processed / 54 dropped) |
| Conformance, real model, tunnel | same with `wss://signlang.dheynorlanda.me/ws` | **9/9 PASS** |
| Server path == offline replay | `python training/server_replay_check.py` (32 test clips over WebSocket) | identical committed events on 32/32; exactly right 23/32 |
| Server speed | `client.py --headless` (dark camera, so only timing is meaningful) | server total p50 22 ms; round trip p50 23 ms local, 73 ms / p95 127 ms through the tunnel |

Not verified: behaviour on a real live webcam with real lighting (the live check, `docs/LIVE_CHECK.md`, is the next step and is the user's); the 100 ms laptop-to-laptop target on the real network (Phase 9); the C++ port of features/decision (Phase 6 parity); a test set with real sentences (the sentence streams are built by joining two clips of the same signer, 33 of them); per-word numbers rest on 12-25 test clips each, so they carry roughly +-10 points of noise.
Placeholders: none left in the fallback server. `training/make_placeholder_model.py` stays only as a smoke-test fixture (`server/models/temporal_placeholder.onnx`, 7 classes, [1,30,233]). `client.py --live-check` is still only a spec idea (use `docs/LIVE_CHECK.md` with the normal client).
New/changed code: `training/{extract_keypoints,signfeat,sign_data,sign_model,train,replay_eval,report_test,export_onnx,server_replay_check}.py`, `server/fallback/{decision,test_decision,fallback_server}.py`, `server/config/server.json`, `protocol/{labels.json,PROTOCOL.md,conformance_test.py}`, `docs/LIVE_CHECK.md`.

## Step 2-4 follow-ups (2026-11-19): history purge, token rotation, disk cleanup
- **History purge DONE.** `git filter-repo --invert-paths --path data/raw --path LAPTOP_SETUP.md` on a fresh clone, force-pushed (`main`: `f7dc5e6` -> `ceafc71`). Verified on a NEW fresh clone: `git log --all -- data/raw LAPTOP_SETUP.md` prints nothing; pack size 122 KB. The mirror backup (`C:\dev\backup-before-purge.git`) was deleted after that check. This clone was reset with `git fetch` + `git reset --hard origin/main`; gitignored data, models, engines and runs untouched. Laptop: `git fetch origin` then `git reset --hard origin/main` (NOT pull). Old clones/forks and GitHub's cached SHAs may still show the old commits; the repo is private.
- **Token rotated:** new random 256-bit `SIGNLANG_TOKEN` set with `setx` on this PC (never written to a file); the laptop needs the same value (`LAPTOP_SETUP.md`, step 4). `LAPTOP_SETUP.md` was recreated WITHOUT any secret and is left uncommitted for the user. The old token (in the purged history and any old clone) no longer works.
- **Disk cleanup DONE:** free space on C: 87.3 GB -> 130.4 GB. Deleted: `C:\dev\Datasets\ASL_Citizen.zip` (42.77 GB), `C:\dev\backup-before-purge.git` (0.01 GB), 16 non-deployed seed checkpoints `training/runs/*/seed*.pt` (0.08 GB), `data/public/asl_citizen/phase3/*.bin` crop caches (~0.3 GB, regenerable with phase3_eval.py). Kept: extracted subset (videos 2.66 GB), keypoint cache, deployed `best.pt` + ONNX + engines, `server/models/candidates` (the keypoint engines were built from them). NOTE: re-running `replay_eval.py --tune` needs `seed*.pt`, i.e. re-running `train.py --seeds 8` (~1 min per seed per model).

## Phase 4: C++ server core: DONE (2026-11-19)
`server/cpp` (CMake + Ninja + vcpkg manifest; build with `server\scripts\build_cpp.ps1`). WebSocket/HTTP on 127.0.0.1:8765, token auth (HTTP 401 before upgrade), `/health`, latest-frame slot, startup checks (sm_120, VRAM >= 4 GB free), preallocation, nvJPEG decode, error replies. It already returns REAL results (the full pipeline of Phase 6 was wired in the same step).

| Check | Command | Result |
|---|---|---|
| Conformance, C++ server, localhost | `client\.venv\Scripts\python protocol\conformance_test.py --url ws://127.0.0.1:8765/ws` | **9/9 PASS** (rapid fire: sent 60, processed 2, dropped 58) |
| Conformance, C++ server, tunnel | same with `wss://signlang.dheynorlanda.me/ws` | **9/9 PASS** (sent 60, processed 60, dropped 0) |
| Client through the tunnel | `client.py --headless --duration 10 --width 640 --height 480` | 94 frames sent, 94 results, no drops; round trip p50 21.3 ms / p95 25.8 ms, server total p50 4.7 ms (dark camera: only timing is meaningful) |
| Python fallback | unchanged, still 9/9 (it must be stopped to free port 8765 while the C++ server runs) | |

## Phase 5: kernels: DONE (2026-11-19) and Phase 6: full C++ pipeline: DONE (2026-11-19)
Pipeline: nvJPEG -> fused crop/normalise kernel (RTMW-x 384x288 full frame) -> TensorRT FP16 -> SimCC decode kernel -> features/decision in C++ -> SignNet TensorRT FP16. Design choices: DECISIONS "Phase 4-6".

| Check | Command | Result |
|---|---|---|
| C++ unit tests | `server\cpp\build\release\signlang_tests.exe` | **33/33 pass**: 20 decision cases (the same 20 as `test_decision.py`), 4 feature tests (window features vs Python golden to 2e-5, window selection by timestamp), 6 kernel tests, 3 protocol tests |
| Kernel tests vs references | in `signlang_tests` | preprocess == CPU reference (>99.9% bit-equal, max 1 grey level), preprocess vs real `cv2.warpAffine` + normalisation within 2 grey levels, SimCC decode == numpy `simcc_decode` incl. ties and score<=0, full-size (133 x 576/768) vs CPU loops |
| compute-sanitizer | `server\scripts\sanitize.ps1 -Jpegs <dir>` | memcheck 0 errors, 0 bytes leaked; racecheck, initcheck, synccheck 0 errors (all on the 33 tests); memcheck on the FULL pipeline incl. TensorRT (4 real frames): 0 errors. Needs `--launch-timeout 0` on this driver (default attach timeout hangs) |
| **Keypoint parity** (24 test clips x 3 frames = 72 live-format frames) | `training\.venv\Scripts\python server\cpp\tests\parity_keypoints.py` | C++ (nvJPEG + kernel + TensorRT FP16) vs Python (cv2 + ORT FP32), pixel error in the 640x480 frame: all 133 points p50 0.00 / p95 1.96 / p99 8.7 / max 799 px; confident points (score >= 1.9) p50 0.00 / p95 1.96 / p99 3.1 / max 403 px; the 57 points the model uses p50 0.00 / p95 3.1 / p99 15 px; flags 1 of 360 differ. 1.96 px = one SimCC bin (0.5 model px x scale 3.9): the rare large errors are argmax flips between two distant peaks on ambiguous (low-confidence) points. First-frame network input differs by mean 0.07 grey levels (p99 2, max 9; includes nvJPEG vs libjpeg decoding) |
| **Server events vs offline replay** (32 test clips through the running C++ server) | `training\.venv\Scripts\python training\server_replay_check.py` | identical committed events on **32/32**; exactly right 23/32 (same as the Python server and the offline replay) |
| Per-stage latency, `--bench 500` on a 640x480 frame | `signlang_server.exe --bench 500 frame.jpg` | decode p50 0.30 / p95 0.44 ms; keypoints (preprocess + RTMW-x TensorRT + decode + copy) p50 3.76 / p95 5.26 / p99 7.83 ms; temporal (features + SignNet) p50 0.59 / p95 0.89 ms; **total p50 4.71 / p95 6.43 / p99 9.48 ms** (targets: decode ~3, keypoints ~12, temporal ~1, total ~20 ms: all met). Python server: ~22 ms |
| Kernel times (Nsight Systems, 323 frames) | `nsys profile ... --bench 300` | fused preprocess 2.8 us median (target < 500 us), SimCC decode 3.4 us; the rest of the GPU time is TensorRT's own kernels |

Not verified / not done:
- **Nsight Compute numbers for the preprocess kernel: blocked by Windows permissions.** `ncu` fails with `ERR_NVGPUCTRPERM` (no access to GPU performance counters). USER ACTION if wanted: NVIDIA Control Panel -> Desktop -> Enable Developer Settings -> Manage GPU Performance Counters -> "Allow access ... to all users", then rerun `ncu --kernel-name preprocess_kernel ...`. Nsight Systems durations above replace it for now; the kernel is ~3 us, far under budget.
- Live end-to-end view with real keypoints drawn on the client (a debug keypoint field behind a flag) is NOT implemented; keypoint correctness is shown by the parity numbers above instead.
- Phase 9 items not started: CUDA Graph, runtime degrade (tier 1), `--mode safe`, failure injection, 10-minute soak, zero-allocation counter, the 100 ms laptop-to-laptop target on the real network (headless dark-camera run: round trip p50 21 ms through the tunnel). Tiers 2/3 skipped on purpose.
- `parity_check.py` (spec name) is `server/cpp/tests/parity_keypoints.py`; the 5-clip requirement is exceeded (24 clips).
- Placeholders: none in the C++ server. `ixwebsocket` stays in `vcpkg.json` but is not linked (see DECISIONS).

## Hardening round 2 (2026-11-19, after Phase 4-6 were committed)
Rule from the user: no threshold or decision tuning until the live-check table arrives. Nothing below changes model behaviour. The live server (release build, port 8765) was NOT touched; new code is built in `server/cpp/build/dev` and tested on other ports.

### Step 1: network-layer robustness: DONE
Refactor: `src/processor.h` (`IProcessor`, `Result`, `DecodeError`) so the network layer needs no CUDA; CMake now has `signlang_net` (CPU only: net, features, decision, config) and `signlang_core` (GPU). Net layer rewritten for strictness: whole-request handshake deadline (`net.handshake_timeout_ms`, default 5 s: a trickle of bytes cannot hold a thread), 431 for headers > 16 KB, strict request line, `Sec-WebSocket-Version: 13` required (426), connection cap (`net.max_connections`, default 64, then HTTP 503), RAII sockets (the old early-return paths leaked a socket), WebSocket frames strict: client frames must be masked, no reserved bits, only opcodes 0/1/2/8/9/10, control frames <= 125 bytes and unfragmented, continuation only inside a message, 64-bit length top bit 0, anything declared above 64 MB is refused with close 1009 without reading it (16-64 MB is drained and answered `too_large`, as conformance requires), every violation -> close frame 1002/1009 and the connection ends.

| Check | Command | Result |
|---|---|---|
| Network robustness tests (12, real server on localhost + fake processor) | `server\cpp\build\dev\signlang_cpu_tests.exe --gtest_filter=NetRobust.*` | **12/12 pass**: 12 malformed-HTTP cases (garbage, no CRLF, bad request line, wrong method/version, 5 kB of random bytes ...), oversized headers (20 kB -> 431, 200 kB unterminated, 150 headers), slowloris (silent partial request cut off in 0.3 s, one-byte-per-100 ms trickle cut off by the total deadline in 0.84 s), handshake rejections (401, 401, 426, 400), 13 bad WebSocket frames each answered with the right close code (1002 or 1009) and TCP close, fragmented message with an interleaved ping, ping/pong/close handshake, text and malformed messages -> error replies and the connection lives, a 5 MB message -> `too_large` and survives, an announced-5-MB-then-vanish client, 300 rapid connect/disconnects in 5 patterns (partial request, handshake never read, half a frame ...) ending with 0 connections counted, connection cap -> 503. After every abuse `/health` still answers |
| All CPU tests | `signlang_cpu_tests.exe` | 39/39 (20 decision + 4 features + 3 protocol + 12 network) |
| **AddressSanitizer** (host memory: bounds, use-after-free) | `server\scripts\build_cpp.ps1 -Dir asan -Asan` then `server\scripts\sanitize.ps1` | the same 39 tests with `/fsanitize=address`: **0 errors**. ASan proven live: a deliberate heap overflow compiled the same way is reported as `heap-buffer-overflow` |
| compute-sanitizer | same script | GPU tests (6 kernel tests): memcheck 0 errors / 0 bytes leaked, racecheck, initcheck, synccheck 0 errors; memcheck on the full pipeline incl. TensorRT 0 errors. NOTE: compute-sanitizer only watches CUDA calls: on the network tests (no CUDA) it reports "terminated before first instrumented API call", so ASan is the memory check that matters for the hand-written network code |
| Conformance with the new network layer | `signlang_server.exe --port 8766` then `conformance_test.py --url ws://127.0.0.1:8766/ws` | **9/9 PASS** |

### Step 2: low-confidence keypoints: REPORT (nothing changed, no retraining)
- **Python `signfeat` and the C++ port do NOT mask individual low-confidence points.** `frame_row` keeps all 57 raw points. The only masking is per HAND: a hand counts as present when >= 15 of its 21 points score >= the model's threshold (loose flag; strict flag adds a higher threshold and 90% inside the frame). `normalize_window` zeroes all 21 points of a hand whose flag is false, and velocities are zeroed where the flag is false in either frame. Body, nose and face points are always passed through raw (the shoulder flag only controls the window's centre/scale). A point with score <= 0 is set to (-1, -1) by `simcc_decode` and then normalised like any other, not masked. C++ is identical (golden test vs Python to 2e-5).
- Measured on the 72 parity frames (`server/cpp/tests/parity_keypoints.py`): **49 of 3,432 points (1.4%) that reach the model have a score below the threshold and are unmasked** (the spare 6 of 21 points of a present hand, mostly).
- **Can the up-to-799 px flips reach the model? Rarely, and only mildly.** 61 of 9,576 points (0.64%) differ from the Python reference by > 20 px (C++ TensorRT FP16 vs ORT FP32: an argmax flips between two distant SimCC peaks); their Python scores are low (median 1.10, max 3.20; threshold 1.9). The worst flips (up to 799 px) are on points the model never uses (feet 18, other body points 11 such as hips/knees). Only 8 of the 61 are among the model's inputs (0.19% of the 57 x 72 points); the worst of those is 74 px = 0.27 shoulder widths, median 0.15 (features are in shoulder widths). That is far below the size of a sign's motion (~1 unit) and the end-to-end check (32/32 identical events) shows no effect, but it is a train/serve difference (training keypoints came from ORT FP32, serving uses TensorRT FP16).
- **Proposed fix (needs retraining, so NOT done):** mask per point instead of per hand: zero a point's coordinates and velocity when its score is below its threshold (hand points inside a present hand, plus elbows/wrists/nose/face, with a looser threshold for body points) and append a per-point presence bit (or reuse the 57 positions with a mask channel). Training would see masked points through the same rule (plus the existing "hand missing" augmentation), so the model learns to ignore them. Evaluate on val first (window macro-F1 + replay) and only then change the server; revisit after the live-check table.

### Step 3: Nsight Compute: BLOCKED on a Windows setting (user action)
`ncu` still stops with `ERR_NVGPUCTRPERM` although counter access was enabled in the Control Panel: the driver value `RmProfilingAdminOnly` is not present under `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak`, so the change did not take effect (it usually needs a reboot; this shell is not elevated and I did not change system settings). USER ACTION: reboot (or run `server\scripts\profile_kernels.ps1` from an administrator shell), then run `powershell -File server\scripts\profile_kernels.ps1 -Build release`; it prints duration, memory throughput (% of peak), achieved occupancy and the optimisation hints for both kernels. Until then the Nsight Systems durations stand (preprocess 2.8 us, SimCC decode 3.4 us per launch). Rough bound, an estimate and not a measurement: preprocess moves ~0.92 MB in + 1.33 MB out in 2.8 us, i.e. ~0.8 TB/s effective, above the 448 GB/s DRAM peak, so it runs from L2 (the frame was just written by nvJPEG) and is latency-bound at ~110k small threads; nothing to optimise at 3 us of a 4.7 ms frame.

### Step 4: Phase 9 essentials: DONE except CUDA Graph (skipped on purpose) and Nsight Compute (user action)
Server changes: tier-1 degrade (primary -> fallback keypoint model + its own SignNet, sticky; triggers: server time > 80 ms for 30 frames, free VRAM < 1 GB checked every 100 frames, or the primary stage throwing; settings in `server.json` `degrade`), `--mode safe` (starts on the fallback profile), `--port N`, `--inject-delay-ms N` (failure injection only), result field `"profile"`. `ixwebsocket` removed from `server/cpp/vcpkg.json` (vcpkg uninstalled it and zlib; the build is still green).

| Check | Command | Result |
|---|---|---|
| **10-minute soak through the tunnel** | `client\.venv\Scripts\python server\scripts\soak.py --minutes 10` (live C++ server, primary) | 4,521/4,521 frames answered, 0 errors, 0 reconnects, 0 server drops; GPU memory (whole GPU) 2245-2284 MiB with no upward trend (spread 39 MiB); server process memory constant 292 MB; round-trip p50 130.1 -> 129.8 ms, server time p50 4.79 -> 4.59 ms: no leak, no creep. The round trip itself (~130 ms in that run) is the internet path: the same test gave 21 / 60 / 133 ms at other moments; loopback is 5.3 ms |
| **Failure injection** (8 cases, real processes) | `client\.venv\Scripts\python server\cpp\tests\failure_injection.py --build release` | **8/8 pass**: missing keypoint engine (exit 1, message names `build_engines.ps1`); missing DEGRADE engine (starts, warns); low VRAM at start (exit 1, "not enough free VRAM"); 14 kinds of odd frames (garbage, truncated, corrupted, PNG-as-JPEG, 1x1, 8x8, grayscale, progressive, 3840x2160 accepted, 6000x4000 refused, 1x2000, zero-filled ...) each answered with a result or an error and the stream continues; 20 clients dying mid-frame (FIN and RST) then a normal client works; server killed and restarted under a running client: the client reconnected by itself (282 of 283 sent frames answered); tier-1 degrade: frames 1-30 on primary, frame 31 on fallback, never back, warning logged; `--mode safe`: profile fallback, conformance 9/9 |
| Full suite on the release build | `server\scripts\run_tests.ps1` | ALL SUITES PASSED: 39 CPU tests, 6 GPU tests, 20 Python decision tests, conformance 9/9 on a throw-away server (port 8771) |
| Per-stage time, release build | `signlang_server.exe --bench 500 training\runs\bench_frame.jpg` | primary: decode 0.32, keypoints 4.02, temporal 0.64, **total 5.09 ms p50** (p95 6.59, p99 7.10); safe/fallback profile: total 4.02 ms p50 (p95 5.23) |
| Switching servers (time until the new server answers `/health`) | `server\scripts\switch_server.ps1 -To cpp / safe / python` | cpp 4.7 s, safe 3.4 s, python 8.7 s |
| Python fallback after my edits | conformance + keypoint field | 9/9; returns `profile` and `keypoints` like the C++ server |

Not done / open in Step 4: **CUDA Graph** (skipped: 5 ms per frame leaves little to gain, per the brief); the VRAM-low and stage-exception degrade triggers exist but only the latency trigger is exercised by a test; restarting the Cloudflare service itself (needs administrator rights, so "tunnel drop" was tested as a server restart under a running client); **laptop-to-laptop round trip: for the user** (`client\.venv\Scripts\python client\client.py --headless --duration 60 --width 640 --height 480`, read the last lines: round trip p50/p95/max and server time; the live window shows them bottom right); the degrade has no automatic way back (by design; restart with `switch_server.ps1 -To cpp`). Nsight Compute: retried again, still `ERR_NVGPUCTRPERM`.
Found and fixed on the way: `put_text` in the client drew outline and fill with different glyph spacing under OpenCV 5 (text looked doubled) -> same-thickness outline at pixel offsets; the start scripts first took `SIGNLANG_TOKEN` from the shell, which after the rotation can be the OLD value (the server would answer 401) -> they now read the Windows user variable first.

### Step 5: demo features: DONE
- **Debug keypoints:** the client sets frame flag bit1; the server (C++ and Python) adds `"keypoints":{"w","h","thr","pts":[[index,x,y,score],...]}` (body 0-12 and both hands 91-132, 55 points, ~1.5 kB) to that reply only. Documented in `protocol/PROTOCOL.md`. Verified: a self-test sent a dataset frame to the live server and drew the client overlay: hand skeleton, face points, shoulders and elbows land in the right places (the image stays in the scratch folder, not in the repo: dataset licence).
- **Client display** (`client/overlay.py`, new): committed word or sentence large in a banner for 4 s ("HOW ARE YOU?", "WHERE ARE YOU?", other words in capitals), small current prediction + confidence bar, latency readout bottom right (`23 ms round trip | server 4.7 ms`), a `SAFE MODE: small keypoint model` tag when the server is on the fallback profile. Keys: **k** keypoints on/off (the server only sends them while on), **d** details, **q**/Esc quit. Not verified: the live window on the laptop's own screen and camera (the rendering was checked on a saved frame).

### Step 6: Phase 10: DONE
New: `server/scripts/{common,start_server,start_fallback,switch_server,run_tests,build_cpp,build_engines,sanitize,profile_kernels,soak}`, `README.md` (PC and laptop setup, repo map), `docs/DEMO_CHECKLIST.md` (start order, switch to the fallback in ~9 s, engine rebuild after a driver update, tunnel/Wi-Fi failure plan including a no-network mode on the PC), `docs/RESULTS.md` (architecture, data/licence, accuracy tables, confusion matrix, end-to-end clip results, per-stage timings, soak, robustness, limitations; measured numbers with their commands, no dataset frames), `LAPTOP_SETUP.md` updated, `docs/LIVE_CHECK.md` updated for the C++ server and `git fetch` + `reset --hard`.
The Phase 10 gate ("a fresh clone builds and runs from the README alone") was NOT tested on a fresh clone: it needs the local-only models, so it can only be checked on this PC; the README steps were written from the commands actually used.

## Pending user actions
1. Commit (file list in the report) and send the laptop live-check table; nothing is tuned until then.
2. Optional: enable GPU performance counters (reboot) and run `server\scripts\profile_kernels.ps1` for the Nsight Compute numbers.
3. Laptop-to-laptop latency run (command above).

### Step 3 update (2026-11-19, later): Nsight Compute measured
Run by the user from an administrator shell (`server\scripts\profile_kernels.ps1`). Preprocess kernel: 3.33 us, memory throughput 31.2% of peak (135 GB/s, DRAM 31.2%, L1/TEX 29.1%, L2 18.8%), compute 27.2%, achieved occupancy 54.0% (theoretical 100%), 192 blocks x 256 threads, 29 registers, 0.89 waves per SM. SimCC decode kernel: 5.22 us, memory 21.8% (95 GB/s), compute 19.9%, achieved occupancy 57.8%, 133 blocks, 24 registers, 0.62 waves. Top limiter for both: the grid is too small to fill the GPU (< 1 wave), i.e. launch/latency bound, not bandwidth or compute bound. Estimated speedups are 10-16% of a few microseconds; nothing optimised (not clearly wrong, ~5 ms per frame). The earlier "runs from L2, above DRAM peak" estimate was imprecise: DRAM is at 31% / 22%, most traffic is served by L2. Step 3 is therefore DONE; the pending user action 2 above is closed.
