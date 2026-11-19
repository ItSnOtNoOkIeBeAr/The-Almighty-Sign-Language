# Demo-day checklist

Paths are relative to `C:\dev\almighty-sign-language` on the PC; commands are PowerShell. Nothing here needs the token printed:
it lives in the Windows user variable `SIGNLANG_TOKEN` (same value on PC and laptop).

## The day before
- [ ] `powershell -File server\scripts\run_tests.ps1` ends with `ALL SUITES PASSED`.
- [ ] A full dress rehearsal from the laptop on the demo network: `client\.venv\Scripts\python client\client.py`, press **k**, sign all six words and both sentences, read the latency bottom right.
- [ ] Laptop: charged, camera works, `git fetch origin` + `git reset --hard origin/main` (never `git pull`), `SIGNLANG_TOKEN` set. PC: plugged in, sleep disabled, Windows updates paused.
- [ ] Know the lighting: bright light on your face and hands from the front, plain background, 0.7-1 m from the camera, both hands in view.

## Start order (PC first, then laptop)
1. Close other GPU programs (browser video, games). The server needs at least 4 GB of free GPU memory and refuses to start otherwise.
2. `Get-Service cloudflared`: Status must be **Running** (the tunnel; do not edit its configuration).
3. `powershell -File server\scripts\switch_server.ps1 -To cpp`
   Expect: `OK: cpp server v0.1.0 answering on 127.0.0.1:8765` and `tunnel: wss://signlang.dheynorlanda.me/ws reaches the cpp server`. It takes about 5 seconds (measured 4.7 s).
4. Laptop: `client\.venv\Scripts\python client\client.py`. Top left must say `connected`. Press **k** to show the hand/body keypoints (the audience sees what the model sees), **d** for details.
5. Sign. Start and end every sign with the hands down. A committed word appears large for 4 seconds. HOW, WHERE and YOU wait up to 1.5 s for a partner word, so they appear slightly late.

## Switch to the Python fallback in under a minute
If the C++ server misbehaves (crash, errors, strange results):

```
powershell -File server\scripts\switch_server.ps1 -To python
```
Stops the C++ server, starts the Python one (same protocol, same models, ONNX Runtime) and checks it through the tunnel. Measured: the Python server answers about 9 seconds after the command (measured 8.7 s). The laptop client reconnects by itself (retry every 1-10 s), no restart needed. Python is slower (~22 ms per frame instead of ~5 ms) but gives the same answers.

Back again: `powershell -File server\scripts\switch_server.ps1 -To cpp` (about 5 seconds (measured 4.7 s)). If the GPU is overloaded but the C++ server is otherwise fine: `-To safe` (small keypoint model, lighter). `-To status` shows what is running; `-To stop` stops everything. The server also drops to the small model by itself when frames get too slow (the client shows `SAFE MODE: small keypoint model`); restart with `-To cpp` to go back.

## After a driver, CUDA or TensorRT update
TensorRT engines are tied to the GPU and the TensorRT version. Symptom: the C++ server exits at start with `engine '...' did not load` and prints the exact rebuild command.
1. Switch to the fallback first so the demo can continue: `switch_server.ps1 -To python` (needs no engines).
2. Rebuild the keypoint engines (about 3 minutes): `powershell -File server\scripts\build_engines.ps1`
3. Rebuild the SignNet engines (seconds): `training\.venv\Scripts\python training\export_onnx.py --model rtmw_xl_384x288 --out server/models/temporal_primary` and the same with `--model rtmw_lm_256x192 --out server/models/temporal_fallback`.
4. If the C++ exe itself stops starting after a CUDA update (missing DLL): `powershell -File server\scripts\build_cpp.ps1`, then `run_tests.ps1`.
5. `switch_server.ps1 -To cpp`.

## If the tunnel or Wi-Fi fails
The client overlay shows `reconnecting in Ns [reason]` and keeps trying; nothing needs restarting on its own. Find out where it breaks:

| Check | Command | Meaning |
|---|---|---|
| Server alive on the PC? | `switch_server.ps1 -To status` and `Invoke-RestMethod http://127.0.0.1:8765/health` | if this fails, restart with `-To cpp` |
| Tunnel up? | `Get-Service cloudflared` and `Invoke-RestMethod https://signlang.dheynorlanda.me/health -UserAgent x` | service stopped: `Restart-Service cloudflared` in an **administrator** shell. Do not change its configuration or token |
| Laptop online? | open any website; `[reason]` in the overlay says `HTTP 401` (token differs), `HTTP 5xx` (tunnel/server down), `TimeoutError`/`OSError` (no network) | 401: set the same `SIGNLANG_TOKEN` on the laptop and open a NEW terminal |

Workarounds, in order:
1. **Wi-Fi weak or slow** (round trip above ~300 ms or "skipped" frames in details): move closer to the access point, or use the phone hotspot. Lighter stream: `client.py --fps 8`. The client already skips frames instead of building a delay.
2. **Internet or tunnel down for good: demo without any network.** Run the client on the PC itself, pointing at the local server, with a webcam on the PC: `client\.venv\Scripts\python client\client.py --url ws://127.0.0.1:8765/ws --width 640 --height 480`. Same overlay, no tunnel involved (the PC needs a camera; a USB webcam works).
3. **GPU trouble:** `switch_server.ps1 -To safe`, or `-To python`.

## Quick diagnosis
| What you see | Likely cause | Fix |
|---|---|---|
| Word never appears | hands not visible / too dark / signing too fast | light from the front, both hands in view, hold the sign ~1 s, hands down between signs; press **k** to see whether the hands are detected |
| Wrong word | the model confuses it (see `docs/RESULTS.md`, `you` and `where` are weakest) | repeat slowly; sign more clearly |
| `SAFE MODE: small keypoint model` | server started in safe mode or degraded itself | `switch_server.ps1 -To cpp` |
| `HTTP 401 (check SIGNLANG_TOKEN)` | token differs between laptop and PC | same value on both, new terminal |
| Round trip high | Wi-Fi / internet; the server itself needs ~5 ms | see the table above; the overlay shows both numbers |
| C++ server will not start: `not enough free VRAM` | another program holds the GPU | close it, start again |
| C++ server will not start: `engine ... did not load` | driver/TensorRT update | section "After a driver update" |
| Port already in use | the other server is still running | `switch_server.ps1 -To status`, then `-To stop` |

## After the demo
`switch_server.ps1 -To stop`. Logs of the last run: `logs\cpp_server.log` / `logs\fallback_server.log` (not in git).
