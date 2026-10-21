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
Phase 0 done. Phase 1 (dependencies) in progress: blocked on the user downloading TensorRT.
Next step: user gives the TensorRT unzip path -> verify version vs release notes -> CMake link test
(CUDA + nvJPEG + TensorRT) -> trtexec FP16 engine -> check `training/.venv` (pip freeze, torch CUDA,
ORT CUDAExecutionProvider) -> write training/requirements.txt -> commit Phase 1 gate.

## Build environment
`cl`/`cmake`/`ninja` are not on the normal PATH. Scripts load the VS dev environment themselves via
`server/scripts/vsenv.ps1` (calls `vcvars64.bat`). Opening VS Code from the
"x64 Native Tools Command Prompt for VS 2022" also works.

## How to build and run
(Filled in as phases land.)
