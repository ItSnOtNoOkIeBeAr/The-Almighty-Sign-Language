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

## Phase 2
- **Port 127.0.0.1:8765 for both servers** (user request, avoids clashes with other projects). BUILD_SPEC.md, PROTOCOL.md and `server.json` updated; port verified free first.
- **Fallback 401 via a pure-ASGI wrapper (`AuthWS`)**: FastAPI closing a WebSocket before `accept` yields HTTP 403, not the required 401. The wrapper answers the `websocket.http.response` denial extension. uvicorn then logs a false "ASGI callable returned without completing handshake" ERROR; that one message is filtered (verified in uvicorn's `websockets_sansio_impl.py`: denial never sets `handshake_complete`, and `send_500_response` returns early).
- **uvicorn `ws_max_size` = 16 MiB, app limit 2 MiB**: a frame over the limit must get a `too_large` reply and keep the connection (PROTOCOL.md); uvicorn's own limit would instead close the socket (code 1009). Anything over 16 MiB still closes the socket, which is acceptable for a clearly abusive client.
- **Placeholder shapes**: temporal model `[1, 45, 96] -> [1, 8]`; the 96 = 8x12 grayscale thumbnail of the dummy keypoint step. Real F is decided in Phase 3/7; `window.feature_dim` in `server.json` must change with it.
- **EMA smoothing in the fallback (alpha 0.5)** and `run_every_k_frames` are read from `server.json` so the C++ server can share them.
- **Client uses `websockets.sync` + threads** (sender+reconnect thread, one receiver thread per connection, capture/display in the main thread): no asyncio in the laptop client, fewer moving parts, and a slow network cannot freeze the preview.
- **Recorder samples like live streaming** (newest camera frame at each 1/fps tick, via a capture thread, repeats allowed): the first version waited for each camera frame and stretched clips to 4.2 s on this dim ~12 fps webcam while `meta.json` said 15 fps. Same `prepare_frame` + `encode_jpeg` as the live path.
- **Extra `meta.json` key `unique_camera_frames`** (beyond spec section 6): lets training later drop clips recorded with a starved camera. The recorder also prints a warning below 80% distinct frames.
- **Token handling**: generated with `RandomNumberGenerator` (32 bytes, base64url, 43 chars), stored with `setx` as a user env var, read back by length/equality only. Never written to a file or echoed. Read it on this PC with `[Environment]::GetEnvironmentVariable('SIGNLANG_TOKEN','User')`.
- **Camera on this PC**: A4tech PC Camera = index 0, 640x480 only. Spec default stays 1280x720 in `client/config.json`; tests here override with `--width 640 --height 480`. Training data must be recorded at one consistent resolution, to be settled before Phase 7.
- **Public hostname `signlang.dheynorlanda.me`** (user's Cloudflare zone, published application route to `http://localhost:8765`). `client/config.json` now defaults to `wss://signlang.dheynorlanda.me/ws`; local tests pass `--url ws://127.0.0.1:8765/ws`. The hostname is not a secret; the tunnel token and `SIGNLANG_TOKEN` are, and are in no file.
- **Conformance health check sends an explicit User-Agent**: Cloudflare blocks the default `Python-urllib` signature (error 1010). Cloudflare settings were left untouched.

## Phase 2 follow-up
- **Client flow control (`max_inflight`, default 3)**: the server's latest-only slot cannot stop a backlog that builds in the laptop's own send buffer / Wi-Fi queue. The client therefore skips a send while more than 3 frames are unanswered (answers include per-frame errors; a 2 s timeout prevents deadlock). Proven with a throttled link: RTT 8.6 s -> 0.25 s. Cost: on a slow link the effective fps drops instead of latency growing, which is the right trade for live recognition.
- **Stream settings are a deliberate choice, not a default**: 1280x720 q80 at 15 fps needs ~10 Mbit/s up, more than this laptop's Wi-Fi gives. Candidates measured: 640x360 q70 (22 KB/frame). Decide together with Phase 3 (hand detail needed by the keypoint model) before recording training data.
