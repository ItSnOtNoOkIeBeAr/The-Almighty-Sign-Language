"""Makes onnxruntime-gpu find its CUDA 13 / cuDNN 9 DLLs on Windows, then returns a session factory.

Why: onnxruntime-gpu 1.31 needs cublasLt64_13.dll (CUDA 13 toolkit, not on PATH) and cuDNN 9
(only shipped inside torch's lib folder here). Without them it SILENTLY falls back to CPU, so
make_session() raises instead of accepting a CPU-only session.
Import this module BEFORE onnxruntime.
"""
import os
from pathlib import Path

_CUDA_BIN = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")) / "bin" / "x64"
_TORCH_LIB = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages" / "torch" / "lib"

for _d in (_CUDA_BIN, _TORCH_LIB):
    if _d.is_dir():
        os.add_dll_directory(str(_d))

import onnxruntime as ort  # noqa: E402


def make_session(model_path: str) -> "ort.InferenceSession":
    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    active = sess.get_providers()
    if active[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"onnxruntime fell back to {active}; CUDA DLLs not found (looked in {_CUDA_BIN}, {_TORCH_LIB})")
    return sess
