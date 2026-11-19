# LAPTOP_SETUP.md: set up the sign-language client on the laptop

The laptop runs only the CLIENT: it captures the webcam, sends JPEG frames to the PC over the tunnel and shows the labels
that come back. The PC does all the AI work. The laptop has no GPU: install only `opencv-python`, `websockets`, `numpy`.

- Server URL (default in `client/config.json`): `wss://signlang.dheynorlanda.me/ws`
- Repo: https://github.com/ItSnOtNoOkIeBeAr/The-Almighty-Sign-Language.git (private: sign in to GitHub)

## Secret rules
The app secret is NEVER in this repo. It lives only in the Windows user environment variable `SIGNLANG_TOKEN`.
Do not write it into any file, commit, chat or log. Do not touch any Cloudflare token or config.

## Steps
1. Python 3.11 or newer: `python --version`.
2. Get the code (path without spaces). First time:
   ```
   git clone https://github.com/ItSnOtNoOkIeBeAr/The-Almighty-Sign-Language.git C:\dev\almighty-sign-language
   ```
   An existing clone made BEFORE 2026-11-19 has the old history (the repo history was rewritten). Do NOT `git pull`:
   ```
   cd C:\dev\almighty-sign-language
   git fetch origin
   git reset --hard origin/main
   ```
3. Client environment:
   ```
   cd C:\dev\almighty-sign-language\client
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
4. Set the token. On the PC, show it in a terminal only you can see (`echo %SIGNLANG_TOKEN%` in cmd), then on the laptop:
   ```
   setx SIGNLANG_TOKEN "<value from the PC>"
   ```
   Open a NEW terminal afterwards (`setx` only affects new terminals). Never paste it into a file or chat.
5. Run: `client\.venv\Scripts\python client\client.py` (add `--width W --height H` if the camera is not 640x480).
   After setup, follow `docs/LAPTOP_LIVE_TEST.md` (preflight, latency run, the live test and results table).
