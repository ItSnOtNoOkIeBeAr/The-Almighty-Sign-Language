# CLAUDE.md: The Almighty Sign Language

Full spec: `governance/BUILD_SPEC.md` (source of truth). Progress: `governance/STATUS.md`. Choices: `governance/DECISIONS.md`.

## Rules (summary of BUILD_SPEC section 0)
1. Verify, never assume. Each phase ends with a gate: run it, show the real output, then mark it done.
2. No silent stubs. Placeholders are labelled `// PLACEHOLDER:` (or `# PLACEHOLDER:`) and listed in STATUS.md.
   No fake tests, no disabled tests.
3. Stop and ask for user-only actions: TensorRT download, tunnel hostname, keypoint model approval, recording data.
4. Fix root causes. Never loosen tolerances or swallow errors to get green.
5. Keep CLAUDE.md, STATUS.md and DECISIONS.md current.
6. THE USER commits and pushes (their request, 2026-10-21). At each passed gate: update STATUS.md, run a secret/size
   scan of `git status`, then tell the user what is ready to commit and suggest a message. Do NOT run `git commit` or
   `git push` unless the user asks in that message. Never include datasets, clips, .onnx, .engine or build folders.
7. Never ask for, print or store the Cloudflare tunnel token. The app secret comes from `SIGNLANG_TOKEN` only.
8. The repo lives at `C:\dev\almighty-sign-language` (no spaces). Quote every path.
9. Small, readable code. Performance rules apply only to the CUDA hot path.

## Fixed facts
- Labels: `protocol/labels.json` (approved): none, hello, what, how, why, where, you; sentences how_are_you / where_are_you are word-joined.
- ASL, public data only (ASL Citizen, license: non-commercial research, never commit/share the data; `data/public/` is gitignored).
- GPU: RTX 5060 Ti, sm_120. CUDA 13.2. MSVC 19.42. CMake 3.29.5 + Ninja (from VS 2022).
- Runtime is C++/CUDA. Python only for training, tests, the client and the fallback server.
- The client runs on a CPU-only laptop: opencv-python, websockets, numpy only.

## Current phase
Phases 0-3, 7, 8 done; 4, 5, 6 done in the C++ server (2026-11-19, see STATUS). Order was 0,1,2,3,7,8, then 4,5,6,9,10 (BUILD_SPEC section 13).
Both servers run the REAL pipeline (RTMW-x 384x288 keypoints -> SignNet -> decision, `--profile fallback` = RTMW-l-m 256x192) and pass conformance 9/9:
`server/cpp` (signlang_server.exe, TensorRT FP16, p50 ~5 ms/frame) is the primary; `server/fallback/fallback_server.py` (ORT, p50 ~22 ms) is the backup.
Only ONE of them can own port 8765 at a time (the tunnel points at it). The user's live check (`docs/LIVE_CHECK.md`) is pending: do not tune anything until the user sends it.
Deadline Nov 28; the C++ path passed parity (Nov 19). Phase 9 essentials (network hardening + ASan, soak, failure injection, tier-1 degrade, `--mode safe`) and
Phase 10 (scripts, README, docs/DEMO_CHECKLIST.md, docs/RESULTS.md) are done (STATUS "Hardening round 2"); CUDA Graph skipped, Nsight Compute measured (kernels are latency-bound, nothing to optimise). Open: laptop-to-laptop latency run, optional per-point keypoint masking (needs retraining).
Same-word guard against double commits implemented in decision.py + decision.cpp (STATUS "Double commits fixed"); the RUNNING release server still has the old code until rebuilt (`build_cpp.ps1`) and restarted. Live check 2026-11-20: hello/how/what/why/where work, YOU never appears. Retraining attempt (STATUS "YOU investigation") did NOT hold on TEST and is NOT deployed. Next: user runs `docs/DIAG_YOU.md` (`start_server.ps1 -Foreground -Diag`, needs `build_cpp.ps1` first) and sends `diag_report.py` output; decide the fix from it.
ASL, public data only (ASL Citizen in `data/public/`, local-only weights, never commit data). Git history was purged on 2026-11-19 (data/raw + LAPTOP_SETUP.md),
the token was rotated and the repo is private; the laptop must `git fetch` + `git reset --hard origin/main` (never pull).
Do not touch the tunnel token/config.
Public URL: wss://signlang.dheynorlanda.me/ws (Windows service `cloudflared` -> http://localhost:8765).

## Tool locations
- TensorRT: `C:\dev\TensorRT-10.16.1.11` (set `TENSORRT_ROOT`; put its `bin` on PATH to run exes). vcpkg: `C:\dev\vcpkg`.
- Python venv (training + fallback): `training\.venv`. Import `ort_cuda` before `onnxruntime`.

## Build environment
`cl`/`cmake`/`ninja` are not on the normal PATH. Scripts load the VS dev environment themselves via
`server/scripts/vsenv.ps1` (calls `vcvars64.bat`). Opening VS Code from the
"x64 Native Tools Command Prompt for VS 2022" also works.

## How to build and run
Server port: 127.0.0.1:8765. Token: user env var SIGNLANG_TOKEN (never in files/chat).
- Which server runs: `powershell -File server\scripts\switch_server.ps1 -To cpp|safe|python|stop|status` (one command; the scripts read the token from the Windows user variable and set the CUDA/TensorRT PATH). While the user may be live-checking, build into another dir (`build_cpp.ps1 -Dir dev`) and test on other ports (`signlang_server.exe --port 8766`); do not restart the live server unless nobody is connected (see `logs\cpp_server.log`).
- C++ server (primary): build `server\scripts\build_cpp.ps1`; engines `server\scripts\build_engines.ps1`; direct run `server\cpp\build\release\signlang_server.exe [--profile fallback | --mode safe] [--port N]`
  Tests: `server\scripts\run_tests.ps1` (CPU 39 + GPU 6 + Python 20 + conformance); sanitizers `server\scripts\sanitize.ps1` (ASan build: `build_cpp.ps1 -Dir asan -Asan`); failure injection `client\.venv\Scripts\python server\cpp\tests\failure_injection.py --build release`; soak `client\.venv\Scripts\python server\scripts\soak.py`; `signlang_server.exe --bench N frame.jpg`; keypoint parity `training\.venv\Scripts\python server\cpp\tests\parity_keypoints.py`
  PowerShell caveats: `Start-Process -Wait` and pipes wait for the hidden server child forever; time/launch servers without `-Wait`; do not name a variable like a parameter (case-insensitive).
- Fallback server (real models, ~20 s to load): `training\.venv\Scripts\python server\fallback\fallback_server.py [--profile fallback]`  (open a NEW terminal after setx)
- Decision tests: `training\.venv\Scripts\python -m unittest server/fallback/test_decision.py -v`; retrain/export: `training/train.py`, `replay_eval.py --tune`, `export_onnx.py` (STATUS Phase 7)
- Conformance: `client\.venv\Scripts\python protocol\conformance_test.py --url ws://127.0.0.1:8765/ws`  (expects 9/9)
- Client: `client\.venv\Scripts\python client\client.py [--headless --duration 10] [--url ws://127.0.0.1:8765/ws for local; config default is the tunnel]`; this PC's camera is 640x480: add `--width 640 --height 480`
- Record: `client.py --record <label> --signer <id>`; review: `client.py --review`

