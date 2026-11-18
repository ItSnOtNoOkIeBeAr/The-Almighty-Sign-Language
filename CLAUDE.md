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
- Labels: `protocol/labels.json` (still the original 8 until the user approves the new list: 0 idle, 1 hello, 2 hi, 3 what, 4 how, 5 why, 6 how_are_you, 7 where_are_you).
  PROPOSED (BUILD_SPEC section 1): idle/other, hello, what, how, why, where, you; sentences are word-joined.
- ASL, public data only (ASL Citizen, license: non-commercial research, never commit/share the data; `data/public/` is gitignored).
- GPU: RTX 5060 Ti, sm_120. CUDA 13.2. MSVC 19.42. CMake 3.29.5 + Ninja (from VS 2022).
- Runtime is C++/CUDA. Python only for training, tests, the client and the fallback server.
- The client runs on a CPU-only laptop: opencv-python, websockets, numpy only.

## Current phase
Phases 0-3, 7, 8 done (order now 0,1,2,3,7,8, then 4,5,6,9,10; see BUILD_SPEC section 13). The Python fallback server runs the REAL pipeline
(RTMW-x 384x288 keypoints -> SignNet -> decision, `--profile fallback` = RTMW-l-m 256x192). GATE: waiting for the user's live check
(`docs/LIVE_CHECK.md`) and approval before starting the C++ server (Phase 4). Labels: none, hello, what, how, why, where, you; sentences
how_are_you / where_are_you are joined from words. ASL, public data only (ASL Citizen in `data/public/`, local-only weights, never commit data).
Pending user decisions: purge `data/raw` clips from git history + make repo private (STATUS Step 1); primary vs fallback keypoints.
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
- Fallback server (real models, ~20 s to load): `training\.venv\Scripts\python server\fallback\fallback_server.py [--profile fallback]`  (open a NEW terminal after setx)
- Decision tests: `training\.venv\Scripts\python -m unittest server/fallback/test_decision.py -v`; retrain/export: `training/train.py`, `replay_eval.py --tune`, `export_onnx.py` (STATUS Phase 7)
- Conformance: `client\.venv\Scripts\python protocol\conformance_test.py --url ws://127.0.0.1:8765/ws`  (expects 9/9)
- Client: `client\.venv\Scripts\python client\client.py [--headless --duration 10] [--url ws://127.0.0.1:8765/ws for local; config default is the tunnel]`; this PC's camera is 640x480: add `--width 640 --height 480`
- Record: `client.py --record <label> --signer <id>`; review: `client.py --review`

