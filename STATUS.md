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
- `cudaRuntimeGetVersion` reports 13.4 although the toolkit is 13.2.51 and `cudart64_13.dll` resolves to the 13.2 folder. Unexplained; harmless so far, will recheck if a kernel/runtime mismatch ever shows up.
- Engines are tied to this TensorRT version and GPU; rebuild after upgrading either.
- `training/.venv` is the shared venv for training and the fallback server. The laptop client will get its own CPU-only requirements in Phase 2.

## Placeholders
None yet.

## Known issues
- `cl`/`cmake` are only available after `vcvars64.bat`; handled by scripts.

