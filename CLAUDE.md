# CLAUDE.md: The Almighty Sign Language

Full spec: `BUILD_SPEC.md` (source of truth). Progress: `STATUS.md`. Choices: `DECISIONS.md`.

## Rules (summary of BUILD_SPEC section 0)
1. Verify, never assume. Each phase ends with a gate: run it, show the real output, then mark it done.
2. No silent stubs. Placeholders are labelled `// PLACEHOLDER:` (or `# PLACEHOLDER:`) and listed in STATUS.md.
   No fake tests, no disabled tests.
3. Stop and ask for user-only actions: TensorRT download, tunnel hostname, keypoint model approval, recording data.
4. Fix root causes. Never loosen tolerances or swallow errors to get green.
5. Keep CLAUDE.md, STATUS.md and DECISIONS.md current.
6. Commit at each passed gate. Never push. Never commit datasets, clips, .onnx, .engine or build folders.
7. Never ask for, print or store the Cloudflare tunnel token. The app secret comes from `SIGNLANG_TOKEN` only.
8. The repo lives at `C:\dev\almighty-sign-language` (no spaces). Quote every path.
9. Small, readable code. Performance rules apply only to the CUDA hot path.

## Fixed facts
- Labels (fixed order): `protocol/labels.json`, 0 idle, 1 hello, 2 hi, 3 what, 4 how, 5 why, 6 how_are_you, 7 where_are_you.
- GPU: RTX 5060 Ti, sm_120. CUDA 13.2. MSVC 19.42. CMake 3.29.5 + Ninja (from VS 2022).
- Runtime is C++/CUDA. Python only for training, tests, the client and the fallback server.
- The client runs on a CPU-only laptop: opencv-python, websockets, numpy only.

## Current phase
Phases 0-1 done. Phase 2 PARTIAL: everything passes locally; BLOCKED on the Cloudflare tunnel (service not installed,
hostname not given). Do not touch the tunnel token/config. After the tunnel gates pass: push, then Phase 3 (keypoint model
selection) needs user approval before Phase 4.

## Tool locations
- TensorRT: `C:\dev\TensorRT-10.16.1.11` (set `TENSORRT_ROOT`; put its `bin` on PATH to run exes). vcpkg: `C:\dev\vcpkg`.
- Python venv (training + fallback): `training\.venv`. Import `ort_cuda` before `onnxruntime`.

## Build environment
`cl`/`cmake`/`ninja` are not on the normal PATH. Scripts load the VS dev environment themselves via
`server/scripts/vsenv.ps1` (calls `vcvars64.bat`). Opening VS Code from the
"x64 Native Tools Command Prompt for VS 2022" also works.

## How to build and run
Server port: 127.0.0.1:8765. Token: user env var SIGNLANG_TOKEN (never in files/chat).
- Fallback server: `training\.venv\Scripts\python server\fallback\fallback_server.py`  (open a NEW terminal after setx)
- Conformance: `client\.venv\Scripts\python protocol\conformance_test.py --url ws://127.0.0.1:8765/ws`  (expects 9/9)
- Client: `client\.venv\Scripts\python client\client.py [--headless --duration 10] [--url wss://<host>/ws]`; this PC's camera is 640x480: add `--width 640 --height 480`
- Record: `client.py --record <label> --signer <id>`; review: `client.py --review`

