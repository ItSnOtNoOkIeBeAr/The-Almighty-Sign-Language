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

## Phase 1: Dependencies: IN PROGRESS
- Done: vcpkg at `C:\dev\vcpkg` (tool 2026-09-26), `vcpkg install --triplet x64-windows` succeeded for
  ixwebsocket 12.0.1 (no SSL), nlohmann-json, spdlog 1.17.0, gtest.
- Done: confirmed ixwebsocket can do 401-before-upgrade via a `WebSocketServer` subclass (see DECISIONS).
- In progress: `training/.venv` (torch cu128, onnx, onnxruntime-gpu, fastapi, uvicorn, websockets, opencv, numpy).
- Blocked: TensorRT not found on disk; the user must download it (NVIDIA login required).
- Not yet verified: CMake link test (CUDA + nvJPEG + TensorRT), `trtexec` FP16 engine build, ORT on CUDAExecutionProvider.

## Placeholders
None yet.

## Known issues
- `cl`/`cmake` are only available after `vcvars64.bat`; handled by scripts.
