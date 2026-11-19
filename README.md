# The Almighty Sign Language

Real-time American Sign Language recognition. A laptop webcam streams JPEG frames over a Cloudflare Tunnel to a PC with an
RTX GPU; the PC finds the hands and upper body, recognises the sign and sends the result back, and the laptop shows it
large on the video.

Vocabulary: **hello, what, how, why, where, you** (plus `none`). The sentences **how are you?** and **where are you?** are
joined from the words (HOW then YOU; WHERE then YOU or YOU then WHERE) within 1.5 s.

```
laptop (CPU only)                          PC (RTX 5060 Ti)
webcam -> JPEG 640x480 q70, 10 fps  --wss-->  Cloudflare Tunnel -> 127.0.0.1:8765
overlay: word, keypoints, latency   <-------  C++/CUDA server: nvJPEG -> RTMW keypoints (TensorRT FP16) -> SignNet -> decision
                                              (Python fallback server: same protocol, same models, ONNX Runtime)
```

- Protocol: [protocol/PROTOCOL.md](protocol/PROTOCOL.md) · Design spec: [governance/BUILD_SPEC.md](governance/BUILD_SPEC.md) · Progress: [governance/STATUS.md](governance/STATUS.md) · Choices: [governance/DECISIONS.md](governance/DECISIONS.md)
- Measured results (every number with its command): [docs/RESULTS.md](docs/RESULTS.md)
- Demo day: [docs/DEMO_CHECKLIST.md](docs/DEMO_CHECKLIST.md) · Live check on the laptop: [docs/LIVE_CHECK.md](docs/LIVE_CHECK.md)

## Data and licence

Recognition models are trained on **ASL Citizen** (Microsoft Research licence: non-commercial research use only). The data
and the trained weights stay on the author's PC: `data/`, `server/models/` and `training/runs/` are gitignored. Never commit,
share or upload them.

## Set up the PC (Windows 11, NVIDIA RTX 50-series, CUDA 13.2)

Everything lives in `C:\dev\almighty-sign-language` (no spaces in the path). Commands are PowerShell, run from the repo root.

1. **Tools** (already installed on the demo PC): Visual Studio 2022 (C++ workload), CUDA 13.2, CMake + Ninja (from VS), TensorRT 10.16 unzipped to `C:\dev\TensorRT-10.16.1.11`, vcpkg in `C:\dev\vcpkg`, Python 3.11.
2. **Python environments:** `python -m venv training\.venv` then `training\.venv\Scripts\pip install -r training\requirements.txt`; `python -m venv client\.venv` then `client\.venv\Scripts\pip install -r client\requirements.txt`.
3. **Models** (not in git; weights are local-only). `server\models\` must hold `rtmw_x_384x288.onnx`, `rtmw_l_m_256x192.onnx`, `temporal_primary.onnx`, `temporal_fallback.onnx` (the temporal ONNX files and `.engine` files come from `training\export_onnx.py`). Then build the keypoint engines: `powershell -File server\scripts\build_engines.ps1`.
4. **Secret:** `setx SIGNLANG_TOKEN <long random value>`, then open a NEW terminal. It is the only place the token lives; never put it in a file or a chat.
5. **Build:** `powershell -File server\scripts\build_cpp.ps1` (first build takes a few minutes).
6. **Check everything:** `powershell -File server\scripts\run_tests.ps1` (C++ tests, Python decision tests, conformance against a throw-away server).
7. **Run:** `powershell -File server\scripts\switch_server.ps1 -To cpp`. The Cloudflare tunnel runs as the Windows service `cloudflared` and forwards `wss://signlang.dheynorlanda.me/ws` to `http://localhost:8765`; do not touch its configuration.

Servers: `switch_server.ps1 -To cpp | safe | python | stop | status` (one at a time owns port 8765; `safe` = C++ server on the small keypoint model; `python` = the Python fallback).

## Set up the laptop (no GPU needed)

See [LAPTOP_SETUP.md](LAPTOP_SETUP.md). In short: install Python 3.11+, `git clone` (or `git fetch` + `git reset --hard origin/main` for an existing clone), create `client\.venv` with `pip install -r client\requirements.txt`, set `SIGNLANG_TOKEN` with `setx` (the same value as on the PC) and run

```
client\.venv\Scripts\python client\client.py
```

Keys in the window: **k** show/hide the detected hand and body keypoints, **d** more details, **q** or Esc quit. The latency (round trip and server time) is shown bottom right; the committed word or sentence appears large for 4 s.
Without a window (timing only): `client.py --headless --duration 30`.

## Repository map

| Folder | Content |
|---|---|
| `server/cpp` | C++/CUDA server (`src`, `kernels`, `tests`), built with CMake + vcpkg |
| `server/fallback` | Python fallback server and the reference decision logic |
| `server/scripts` | start, switch, build, test, sanitizer, soak and profiling scripts |
| `server/config` | `server.json` (ports, models, thresholds, decision settings) |
| `client` | laptop client (CPU only) |
| `training` | keypoint extraction, SignNet training, ONNX/TensorRT export, evaluation |
| `protocol` | wire protocol, labels, conformance test |
| `governance`, `docs` | spec, status, decisions, results, checklists |
