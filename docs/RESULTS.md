# Results: The Almighty Sign Language

For the documentation team. Every number below was measured on this project's PC (RTX 5060 Ti 16 GB, sm_120, CUDA 13.2,
TensorRT 10.16, Windows 11) and is followed by the command that produced it, run from the repository root. Where something was
not measured it says so. No dataset video frames are reproduced here (licence, see section 2). Dates: 2026-11-18 to 2026-11-19.

## 1. What the system does

A laptop webcam streams 640x480 JPEG frames at 10 fps over a Cloudflare Tunnel to a PC. The PC detects hand and upper-body
keypoints, a small neural network classifies a 3-second window of keypoint motion into one of seven classes
(`none, hello, what, how, why, where, you`), a rule-based decision layer turns the stream of predictions into committed words,
and joins two words into the sentences "how are you?" and "where are you?". The result goes back to the laptop and is drawn
over the video.

```
 LAPTOP (CPU only: opencv, websockets, numpy)
   webcam -> resize 640x480 -> JPEG q70 @ 10 fps ---------------------+
   overlay: committed word (large), keypoints (key k), latency        |  wss://signlang.dheynorlanda.me/ws
        ^                                                              v
        |                                         Cloudflare Tunnel (Windows service "cloudflared")
        |                                                              |  http://localhost:8765
        |                                                              v
   JSON result per frame  <--  PC: signlang_server.exe (C++/CUDA)       [ python fallback_server.py: same protocol/models ]
                                 1. HTTP/WebSocket layer (own Winsock code): auth (HTTP 401 before upgrade), /health,
                                    latest-frame slot (a newer frame replaces the waiting one), error replies
                                 2. nvJPEG decode on the GPU
                                 3. fused CUDA kernel: crop (full frame) + resize + RGB + normalise  -> network input
                                 4. RTMW-x 384x288 whole-body keypoints, TensorRT FP16 (133 points)
                                 5. CUDA kernel: SimCC arg-max decode -> keypoints in frame pixels
                                 6. features on the CPU: 57 points (hands, arms, face), shoulder-normalised, velocities, 5 flags
                                 7. SignNet (Conv1D + 3 Transformer blocks, 1.3 M parameters), TensorRT FP16, 30-frame window
                                 8. decision logic (CPU): idle gate, EMA, M agreeing frames, cooldown, word joining
                                 (tier-1 degrade / --mode safe: step 4 uses RTMW-l-m 256x192 and its own SignNet)
```

## 2. Data and licence

- **ASL Citizen** (Microsoft Research). Licence: non-commercial research use only; the data, and anything derived from it that
  could reveal it, must not be redistributed. Therefore `data/` (videos, keypoint caches) and the trained weights are never
  committed or shared; only code is in git. ASL only, public data only (nobody was recorded for this project).
- Official signer-disjoint split: 52 signers, train 35 / val 6 / test 11, 2,731 glosses in the full set.
- Subset used (4,826 clips, 2.86 GB, CRC-verified; `python training/extract_asl_citizen.py`): HELLO 32, HOW1+HOW2 62, WHAT1+WHAT2 63,
  WHY 32, WHERE 31, YOU 31 clips (each from 26-30 signers) and 4,575 clips of 150 random other glosses (class `none`).
  Clips are ~2 s of 30 fps video (median 65 frames), resampled to the live rate (10 fps, 640x480, JPEG q70) before use.
- The sentences are not signs in the dataset: they are produced by joining two recognised words within 1.5 s.

## 3. Accuracy on unseen signers (test split: 11 signers, never used for any choice)

Model: RTMW-x 384x288 keypoints + SignNet, window level (533 windows). Command: `training\.venv\Scripts\python training\report_test.py --model rtmw_xl_384x288` (stored in `training/runs/rtmw_xl_384x288/test_report.json`).

**Accuracy 0.848, macro-F1 0.828.**

| Class | precision | recall (per-class accuracy) | F1 | support |
|---|---|---|---|---|
| none | 0.775 | 0.951 | 0.854 | 224 |
| hello | 0.813 | 0.929 | 0.867 | 42 |
| what | 0.984 | 0.800 | 0.882 | 75 |
| how | 0.984 | 0.813 | 0.891 | 75 |
| why | 0.970 | 0.762 | 0.853 | 42 |
| where | 0.871 | 0.692 | 0.771 | 39 |
| you | 0.870 | 0.556 | 0.678 | 36 |

Confusion matrix (rows = true class, columns = predicted):

| true \ pred | none | hello | what | how | why | where | you |
|---|---|---|---|---|---|---|---|
| none | 213 | 2 | 1 | 1 | 0 | 4 | 3 |
| hello | 3 | 39 | 0 | 0 | 0 | 0 | 0 |
| what | 14 | 0 | 60 | 0 | 1 | 0 | 0 |
| how | 14 | 0 | 0 | 61 | 0 | 0 | 0 |
| why | 6 | 4 | 0 | 0 | 32 | 0 | 0 |
| where | 9 | 3 | 0 | 0 | 0 | 27 | 0 |
| you | 16 | 0 | 0 | 0 | 0 | 0 | 20 |

Reading it: the model is conservative. Almost every error is a real word predicted as `none`; one word is almost never mistaken for another. `you` and `where` (pointing-type signs) are weakest.

Training details (command `python training/train.py --model rtmw_xl_384x288 --seeds 8`): train on the 35 training signers, early stopping and seed/threshold selection on the 6 validation signers, test reported once. Window macro-F1 on validation, mean of 8 seeds: RTMW-x 384x288 0.862, RTMW-l-m 256x192 0.889 (`training/runs/*/metrics.json`).
Both keypoint models were trained; on validation the end-to-end score favoured RTMW-x 384x288, which is the deployed primary.

## 4. End-to-end results on test clips (full pipeline: keypoints -> features -> SignNet -> decision)

Each test clip is replayed as a 10 fps stream with resting frames before and after. A word clip is correct only when exactly its word is committed. Command: `training\.venv\Scripts\python training\replay_eval.py --model rtmw_xl_384x288` (stored in `training/runs/rtmw_xl_384x288/replay.json`). Decision settings: EMA 0.7, threshold 0.5, 4 agreeing frames, cooldown 800 ms, join window 1500 ms.

| Measure | Result |
|---|---|
| Word clips committed exactly right | 73 of 103 (70.9%): hello 13/14, how 15/25, what 18/25, why 10/14, where 9/13, you 8/12 |
| False triggers on 400 clips of other signs | 15 (3.75%) |
| Sentences (33 streams built by joining two clips of one signer) | 8 of 33 (24%): how_are_you 3/11, where_are_you 5/22 |

Per-word figures rest on 12-25 clips each, so they carry roughly +-10 points of noise. The sentence streams are synthetic joins, not real recordings.
The fallback keypoint model (RTMW-l-m 256x192): words 70/103, false triggers 26/400, sentences 5/33 (`... --model rtmw_lm_256x192`).

**The C++ server gives the same answers as the Python reference.** 32 test clips sent through the running C++ server over a WebSocket produced identical committed events to the offline replay on 32/32 clips (exactly right on 23/32, same as the Python server). Command: `training\.venv\Scripts\python training\server_replay_check.py`.

## 5. Keypoint parity: C++/TensorRT FP16 vs Python/ONNX Runtime FP32

72 live-format frames from 24 test clips. Command: `training\.venv\Scripts\python server\cpp\tests\parity_keypoints.py`. Errors in pixels of the 640x480 frame:

| Points | median | p95 | p99 | max |
|---|---|---|---|---|
| all 133 keypoints | 0.00 | 1.96 | 8.7 | 799 |
| confident (score >= 1.9) | 0.00 | 1.96 | 3.1 | 403 |
| the 57 points the model uses | 0.00 | 3.1 | 15 | 799 |

1.96 px is one SimCC bin. 61 of 9,576 points (0.64%) differ by more than 20 px (arg-max flips between two distant peaks, low confidence); only 8 of them (0.19% of the model's points) reach the model, the worst by 0.27 shoulder widths. Five-flag presence masks differ in 1 of 360 cases.

## 6. Speed

All on the PC, 640x480 frames. Per-stage server time, `signlang_server.exe --bench 500 frame.jpg` (50th / 95th / 99th percentile, milliseconds):

| Stage | primary (RTMW-x 384x288) p50 / p95 / p99 | safe mode (RTMW-l-m 256x192) p50 / p95 / p99 |
|---|---|---|
| nvJPEG decode | 0.32 / 0.53 / 0.76 | 0.29 / 0.45 / 0.56 |
| preprocess kernel + keypoint network (TensorRT FP16) + SimCC decode kernel + copy | 4.02 / 5.30 / 5.67 | 3.09 / 4.08 / 5.92 |
| features + SignNet (TensorRT FP16) + softmax | 0.64 / 0.95 / 1.15 | 0.57 / 0.80 / 1.44 |
| **total per frame** | **5.09 / 6.59 / 7.10** | **4.02 / 5.23 / 8.08** |

Targets from the spec: decode ~3 ms, keypoints ~12 ms, temporal ~1 ms, total ~20 ms; all met. Commands: `signlang_server.exe --bench 500 training\runs\bench_frame.jpg` and `... --profile fallback --bench 500 ...` (one 640x480 live-format frame, 500 frames after 20 warm-up). VRAM in use by the server about 0.26 GB (both profiles loaded; GPU total in use incl. other programs 2.2 GB).

Single kernels (Nsight Systems, `nsys profile ... signlang_server.exe --bench 300 frame.jpg`, 323 launches): fused preprocess 2.8 us median (spec target < 500 us), SimCC decode 3.4 us. The rest of the GPU time is TensorRT's own kernels.
Nsight Compute (memory throughput, occupancy): **not measured**: Windows blocks GPU performance counters for non-administrators (`ERR_NVGPUCTRPERM`). Run `server\scripts\profile_kernels.ps1` after enabling counter access (instructions in the script header); it prints duration, memory throughput (% of peak), achieved occupancy and optimisation hints. Estimate only, not a measurement: the preprocess kernel moves about 0.9 MB in and 1.3 MB out in 2.8 us, above the 448 GB/s DRAM peak, so it runs from the L2 cache and is latency-bound.
Reference: the Python fallback server needs about 22 ms per frame (`client.py --headless`, "server total ms").
Export checks: SignNet ONNX vs PyTorch max logit difference 9e-6 (`training/export_onnx.py`); TensorRT FP16 vs ONNX Runtime 0.004-0.011, 0.475 ms GPU time.

## 7. Soak and latency

**10-minute soak through the tunnel** (`client\.venv\Scripts\python server\scripts\soak.py --minutes 10`, 640x480 JPEG q70 at 10 fps, `wss://signlang.dheynorlanda.me/ws`, C++ server, primary profile):

| Minute | frames answered | round trip p50 / p95 / max (ms) | server time p50 / p95 (ms) | errors | GPU memory in use (MiB) | server process memory (MB) |
|---|---|---|---|---|---|---|
| 1 | 452 | 130.1 / 138.7 / 168 | 4.79 / 10.13 | 0 | 2284 | 292 |
| 2 | 452 | 130.0 / 140.0 / 320 | 4.78 / 6.51 | 0 | 2266 | 292 |
| 3 | 456 | 129.9 / 137.8 / 284 | 4.73 / 6.96 | 0 | 2253 | 292 |
| 4 | 450 | 130.0 / 139.4 / 358 | 4.57 / 7.84 | 0 | 2272 | 292 |
| 5 | 445 | 130.1 / 139.3 / 754 | 4.57 / 9.12 | 0 | 2253 | 292 |
| 6 | 454 | 129.7 / 137.3 / 366 | 4.58 / 9.93 | 0 | 2245 | 292 |
| 7 | 454 | 129.6 / 137.2 / 447 | 4.57 / 9.84 | 0 | 2245 | 292 |
| 8 | 453 | 129.8 / 137.7 / 451 | 4.58 / 6.81 | 0 | 2245 | 292 |
| 9 | 451 | 129.9 / 140.1 / 280 | 4.58 / 8.73 | 0 | 2245 | 292 |
| 10 | 454 | 129.8 / 139.2 / 226 | 4.59 / 9.80 | 0 | 2245 | 292 |

4,521 of 4,521 frames answered, 0 errors or reconnects, 0 server-side drops. GPU memory (whole GPU, all programs) varied 2245-2284 MiB with no upward trend; server process memory constant at 292 MB; round-trip p50 130.1 -> 129.8 ms and server time 4.79 -> 4.59 ms from first to last minute: **no leak and no latency creep**.

**Round trip through the tunnel from this PC is dominated by the internet path, not by the server.** Server time is ~4.6 ms of the ~130 ms in that run. The same test at other moments gave 21 ms (headless client, earlier the same day), 60 ms (12-second soak) and 133 ms (headless client, minutes later); `/health` through the tunnel took 71-234 ms in a row of 5 requests. Loopback round trip is 5.3 ms (`soak.py --url ws://127.0.0.1:8765/ws`). These numbers include only PC-to-Cloudflare-to-PC; they do not include a laptop, its Wi-Fi, its camera or its display.

**Laptop-to-laptop latency: not measured yet.** On the laptop run `client\.venv\Scripts\python client\client.py --headless --duration 60 --width 640 --height 480`; the last lines print the round trip (p50 / p95 / max, send -> result received) and the server time. In the live window the same numbers are bottom right. The camera's own capture delay and the display delay are not included in that figure.

## 8. Robustness (all automated)

| What | Result | Command |
|---|---|---|
| Protocol conformance, C++ server, local and through the tunnel | 9/9 and 9/9 | `client\.venv\Scripts\python protocol\conformance_test.py --url ws://127.0.0.1:8765/ws` (and the `wss://` URL) |
| C++ unit tests | 45/45 (39 CPU: 20 decision, 4 features, 3 protocol, 12 network; 6 GPU kernel tests); plus 20 Python decision tests | `server\scripts\run_tests.ps1` |
| Network layer abuse tests (malformed HTTP, oversized headers, slowloris, bad WebSocket frames, fragmentation, ping/pong/close, 300 rapid connect/disconnects, connection cap) | 12/12 pass; /health still answers after each | `server\cpp\build\release\signlang_cpu_tests.exe --gtest_filter=NetRobust.*` |
| AddressSanitizer on the CPU tests (network layer, features, decision) | 0 errors (and a deliberate overflow is detected) | `server\scripts\build_cpp.ps1 -Dir asan -Asan` then `server\scripts\sanitize.ps1` |
| compute-sanitizer memcheck, racecheck, initcheck, synccheck on the GPU kernel tests; memcheck on the full TensorRT pipeline | 0 errors, 0 bytes leaked | `server\scripts\sanitize.ps1 -Jpegs <folder with jpgs>` |
| Failure injection (missing engine, missing degrade engine, low VRAM at start, 14 kinds of odd/garbage frames, 20 clients dying mid-frame, server restart under a running client, tier-1 degrade, safe mode) | 8/8 pass | `client\.venv\Scripts\python server\cpp\tests\failure_injection.py --build release` |

Tier-1 degrade: if the server's time per frame stays above 80 ms for 30 frames, or free VRAM drops below 1 GB, the server switches by itself to the small keypoint model (RTMW-l-m 256x192) and its SignNet; streams restart on it and the result carries `"profile":"fallback"`. It does not switch back by itself. `--mode safe` starts in that state.

## 9. Known limitations (honest list)

- **Short signs are often missed.** Many dataset signs last 0.5-0.8 s (5-8 frames at 10 fps) and the model's confidence crosses the threshold only in the last frames. A signer holding a sign for about a second should do better than the dataset clips; the live check on a real webcam (`docs/LIVE_CHECK.md`) is the real test and its result is not part of this document yet.
- **`you` (recall 0.56) and `where` (0.69) are weak;** per-word clip figures rest on 12-25 clips (+-10 points).
- **Sentences work for 24% of synthetic joins.** `how`, `where` and `you` wait up to 1.5 s for a partner word, so they appear up to 1.5 s late.
- Trained on isolated-sign studio-style webcam clips of 52 signers; lighting, background and signing style outside that range will hurt. Left-handed signers are not modelled specially. ASL only, 6 words.
- Individual low-confidence keypoints are not masked (only whole hands are): see `governance/STATUS.md`, "Step 2". Fixing it needs retraining and was deliberately not done before the live check.
- Not measured: laptop-to-laptop latency on a real network in demo conditions (the client prints it; see `docs/DEMO_CHECKLIST.md`); the live-check accuracy; Nsight Compute counters unless filled in above; CUDA Graph (not implemented: ~4.7 ms per frame leaves little to gain).
- Tunnel drop: reconnection is tested by restarting the server under a running client; restarting the Cloudflare service itself needs administrator rights and was not done.
