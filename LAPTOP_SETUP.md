# LAPTOP_SETUP.md: set up the sign-language client on this laptop

**For Claude Code running on the laptop.** Read this whole file, then do the steps in order. Run each command,
show the real output, and stop and tell the user if a step fails. Do not guess or work around errors silently.

## What this laptop is for
This laptop runs the CLIENT of "The Almighty Sign Language": it captures the webcam, sends JPEG frames over the
internet to the user's PC, and shows the label the PC sends back. The PC does all the AI work. The laptop has NO GPU
and must NOT install torch, onnxruntime, CUDA or anything heavy. Only `opencv-python`, `websockets` and `numpy`.

- Server URL (already the default in `client/config.json`): `wss://signlang.dheynorlanda.me/ws`
- Repo: https://github.com/ItSnOtNoOkIeBeAr/The-Almighty-Sign-Language.git (may be private: the user may need to sign in)

## Secrets: rules for you
- The app secret is in the section "The token" below. It is a password. Do NOT print it again, do NOT write it into any
  file in the repo, do NOT commit it, and do NOT include it in any message or summary you write.
- You will store it ONLY as a Windows user environment variable named `SIGNLANG_TOKEN` (step 4).
- Never ask the user for, look for, or touch any Cloudflare token or config. This laptop does not need one.
- Do not run `git commit` or `git push`. The user does that themselves.

## Steps

### 1. Check Python (needs 3.11 or newer)
```
python --version
```
If it is missing or older than 3.11, tell the user to install Python 3.11+ from python.org with "Add python.exe to PATH"
ticked, then continue after they confirm. (The pinned numpy needs 3.11+.)

### 2. Get the code into a path WITHOUT spaces
Preferred location: `C:\dev\almighty-sign-language`.
```
git --version
git clone https://github.com/ItSnOtNoOkIeBeAr/The-Almighty-Sign-Language.git C:\dev\almighty-sign-language
```
If git is missing or the clone is refused (private repo), tell the user: either install git / sign in to GitHub, or copy
the `client` and `protocol` folders from the PC by USB into `C:\dev\almighty-sign-language\`. Then continue.

### 3. Create the CPU-only client environment
```
cd C:\dev\almighty-sign-language\client
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip list
```
Expected: `numpy`, `opencv-python`, `websockets` (plus pip/setuptools). Prove there is nothing heavy:
```
.venv\Scripts\python.exe -c "import importlib.util as u; print({m: u.find_spec(m) is not None for m in ('torch','onnxruntime','tensorrt')})"
```
Every value must be `False`.

### 4. Store the token as a user environment variable
Run this (it prints nothing secret). Use the value from the section "The token" below:
```
setx SIGNLANG_TOKEN "<THE TOKEN>"
```
`setx` only affects NEW terminals. For the rest of this session, also set it in the current PowerShell:
```
$env:SIGNLANG_TOKEN = "<THE TOKEN>"
```
Verify WITHOUT printing it:
```
"token set: $([bool]$env:SIGNLANG_TOKEN), length: $($env:SIGNLANG_TOKEN.Length)"
```
Expected: `token set: True, length: 43`.

### 5. Check the server is reachable (needs the user's PC to be on)
```
Invoke-WebRequest https://signlang.dheynorlanda.me/health -UseBasicParsing
```
- HTTP 200 with `{"ok":true,"server":"fallback",...}`: the PC server is up. Continue.
- HTTP 502/530 or a timeout: the PC server is not running or the PC is asleep. Ask the user to start it on the PC
  (`training\.venv\Scripts\python server\fallback\fallback_server.py`), then retry.

### 6. Headless test (no window), 10 seconds
```
cd C:\dev\almighty-sign-language\client
.venv\Scripts\python.exe client.py --headless --duration 10
```
Show the last lines. A good result: `state: connected`, `frames sent` roughly equal to `results`, and a
`round-trip ms: p50 ... p95 ...` line. The label will always be `idle`: the PC model is a random placeholder for now.

Troubleshooting:
- `HTTP 401 (check SIGNLANG_TOKEN)`: token wrong, or this terminal was opened before `setx`. Re-run step 4.
- `cannot open camera`: another app (Zoom, Teams, Discord, the Camera app) holds it. Close it, or try `--camera 1`.
- Stuck on `reconnecting`: server not running (step 5) or no internet.

### 7. Live window test
```
.venv\Scripts\python.exe client.py
```
Expected: a window with the camera view and an overlay: connection state, label, confidence bar, committed history,
round-trip ms, server ms, FPS. Press `q` or Esc to quit. Ask the user to confirm they can see the overlay.

### 8. Optional: record one test clip (tests record mode on this laptop)
```
.venv\Scripts\python.exe client.py --record hello --signer me
```
A countdown runs, then a 3-second clip is saved under `data\raw\hello\me_<timestamp>\` (45 JPEGs + meta.json).
Tell the user the folder path so they can copy it to the PC. Never commit it (it is gitignored).

## Report back to the user (summary only, no secrets)
- Python version, camera resolution if known, whether each step passed
- The headless summary: `frames sent`, `results`, `round-trip p50/p95/max`
- Any warning or error text, exactly as shown

## The token
Value for `SIGNLANG_TOKEN` (43 characters, URL-safe). Treat it as a password:

`uTan7nJeKfCBcpnDadDLwj8PzTc4URhEPy49Fo20uh0`

**User: delete this file from the laptop after setup.** It contains the token in plain text.