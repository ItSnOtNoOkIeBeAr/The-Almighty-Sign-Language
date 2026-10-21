# DECISIONS

One line per non-obvious choice: what and why.

## Phase 0
- **Repo location `C:\dev\almighty-sign-language`**: the user's home path contains a space, which breaks some CUDA/CMake tooling (spec rule 8).
- **VS dev environment loaded by script, not by reopening VS Code**: `cl` and `cmake` are not on the default PATH. Scripts call `vcvars64.bat` (located via the VS install path) so builds work from any terminal. Verified: `cl` 19.42.34436, CMake 3.29.5-msvc4, Ninja from VS.
- **Extra error codes in PROTOCOL.md** (`bad_type`, `too_short`, `bad_version`, `empty_payload`, `internal`): the spec lists `bad_magic|too_large|decode_failed|...`; these fill in the `...` so both servers and the conformance test agree on exact codes.
- **End-of-stream semantics**: spec defines the flag but not the behaviour. Chosen: an EOS message with an empty payload gets no result and resets that connection's decision state.
- **Max message size 2 MB = 2 097 152 bytes, header included**.

## Phase 1
- **vcpkg baseline `857036234d15fbed8f30fed66e898d7725446c0c`** (vcpkg tool 2026-09-26): pinned for reproducible builds. Installed: ixwebsocket 12.0.1, nlohmann-json, spdlog 1.17.0 (fmt 12.2.0), gtest. Triplet `x64-windows`.
- **ixwebsocket without SSL** (`default-features: false`): TLS terminates in `cloudflared` on the same PC, and the server binds to 127.0.0.1 only, so mbedtls would be dead weight.
- **ixwebsocket kept over Boost.Beast, with a custom server subclass**: `ix::HttpServer` is `final` and upgrades WebSocket requests *without* calling the HTTP callback (verified in `IXHttpServer.cpp:95-108`), so it cannot do "401 before upgrade". Instead we subclass `ix::WebSocketServer`, override its virtual `handleConnection`, call `ix::Http::parseRequest`, answer `/health` and 401 ourselves, and call the protected `handleUpgrade(socket, state, request)` only for authorised `/ws` upgrades. Small, contained, and no second networking library.
- **TensorRT 10.16.1.11 (not 11)**: newest 10.x GA; the zip for CUDA 13.0-13.2 matches the 13.2 toolkit and contains sm_120 builder resources. TensorRT 11 is a new major version with possible API changes, so it is held back unless 10.x fails.
- **ORT CUDA DLLs registered by `training/ort_cuda.py`**: onnxruntime-gpu 1.31 needs `cublasLt64_13.dll` (in `CUDA\v13.2\bin\x64`, not on PATH) and cuDNN 9 (taken from torch's lib folder, a CUDA 12.8 build, which works with ORT's CUDA 13 provider here). Without them ORT silently used CPU, so `make_session()` raises unless CUDAExecutionProvider is first.
- **No separate cuDNN install**: TensorRT 10 does not need it, and torch's copy covers ORT.
