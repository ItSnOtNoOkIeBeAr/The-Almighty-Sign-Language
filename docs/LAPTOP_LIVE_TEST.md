# Laptop live test: one page, start to finish

Do `LAPTOP_SETUP.md` first (clone, client environment, token). This page starts where that one ends: it checks that
the laptop is ready, then walks through the live check in a fixed order. Commands are PowerShell, run from `C:\dev\almighty-sign-language`.
Nothing is recorded or saved; the laptop only sends frames and shows the answers.

## 0. The PC must be ready (ask whoever runs the PC)
- `https://signlang.dheynorlanda.me/health` answers `{"ok":true,"server":"cpp",...}` (open it in the laptop's browser).
- The server and the tunnel are both running on the PC. Nobody else is using the same port.

## 1. Preflight on the laptop (5 checks, about 2 minutes)

| # | Check | Command | Good result |
|---|-------|---------|-------------|
| 1 | Python | `python --version` | 3.11 or newer |
| 2 | Code is current | `git fetch origin; git status -sb` | `## main...origin/main`, no `behind`. If behind or odd: `git reset --hard origin/main` (never `git pull`) |
| 3 | Client packages | `client\.venv\Scripts\python -c "import cv2, websockets, numpy; print('ok')"` | `ok` |
| 4 | Token is set (value is not shown) | `if ($env:SIGNLANG_TOKEN) { 'token set' } else { 'token MISSING' }` | `token set`. If missing: `setx SIGNLANG_TOKEN "<value from the PC>"`, then open a NEW terminal |
| 5 | Tunnel reachable | `Invoke-RestMethod https://signlang.dheynorlanda.me/health -UserAgent x` | `ok : True`, `server : cpp` |

If check 5 fails: the PC server or tunnel is down, or the laptop has no internet. Stop and tell the PC side.

## 2. Camera and connection test (headless, 10 seconds, no window)
```
client\.venv\Scripts\python client\client.py --headless --duration 10 --width 640 --height 480
```
Read the last lines:
- frames were sent and answered, with a round trip in ms: **good**.
- `HTTP 401`: the laptop's `SIGNLANG_TOKEN` differs from the PC's. Fix the value, open a new terminal.
- `HTTP 5xx`, `TimeoutError` or `OSError`: tunnel or network problem (try a phone hotspot).
- camera error: close other programs that use the camera (Teams, Zoom, browser tabs), or the camera index is wrong.
  If the picture looks squashed or stretched, pass the camera's real size with `--width W --height H`.

## 3. Latency run (60 seconds, send the last lines back)
```
client\.venv\Scripts\python client\client.py --headless --duration 60 --width 640 --height 480
```
Keep the last lines (round trip p50 / p95 / max and server time). Round trip above about 300 ms: move closer to the
access point or use the hotspot; `--fps 8` makes the stream lighter.

## 3b. Camera speed and send rate (smoother skeleton)
The skeleton is as smooth as the frames you send; the server tracks every frame and still recognises at exactly 10 per second.
1. What does this laptop's camera really deliver? (do it in the light you will demo in; a camera in dim light slows itself down)
   ```
   client\.venv\Scripts\python client\client.py --probe-camera --width 640 --height 480
   ```
   It tries the default, an explicit 30 fps request and MJPG + 30 fps, and prints what each really delivers. If a variant is clearly faster it prints the flags to use (for example `--camera-fps 30 --fourcc MJPG`).
2. Which send rate does your Wi-Fi carry? (needs the server running with the new build; 20 s per rate)
   ```
   client\.venv\Scripts\python client\client.py --sweep 10,15,20,30 --duration 20 --width 640 --height 480 --keypoints
   ```
   Add the flags from step 1 if it found some. The table shows, per rate: camera fps, frames sent and answered, skipped by flow control, dropped by the server, round trip p50 / p95 / max and server time, then the **recommended rate** (highest with round trip p50 <= 100 ms, at most 5% skipped or dropped, and a camera that keeps up).
3. Make it the default: set `"fps"` in `client/config.json` (or pass `--fps N`). Send me the table.

## 4. Open the live window
```
client\.venv\Scripts\python client\client.py --width 640 --height 480
```
- Top left must say `connected`.
- Press **k**: the hand and body keypoints appear on the video. Both hands should show a full skeleton when raised. If one
  hand has no skeleton, light or position is the problem, not the model.
- Press **d** for details (confidence, state, skipped frames). **q** or Esc quits.
- A `SAFE MODE: small keypoint model` tag means the PC switched to the small model; note it and tell the PC side.

## 5. Set up the scene (do this before counting anything)
- Light from the front (window or lamp behind the screen), not behind you.
- 0.7-1 m from the camera: face, shoulders and both hands in the frame. Plain background, nobody else visible.
- Hands start and end every try **down, below the shoulders, out of frame**.
- Sign one word at a time, clear and held about 1 second.

Do 3 practice signs of HELLO first (not counted) to see the whole cycle: hands up, word appears large for 4 s, hands down.

## 6. The test (keep the window open the whole time)
For each try: hands down, sign once, hands down, wait for the committed word (up to 2 s for HOW / WHERE / YOU: they wait
for a partner word), tick one box, repeat.

| Word | Tries | Correct | Wrong word (which?) | Nothing committed |
|------|-------|---------|---------------------|-------------------|
| hello | 10 | | | |
| what | 10 | | | |
| how | 10 | | | |
| why | 10 | | | |
| where | 10 | | | |
| you | 10 | | | |
| how_are_you (HOW, then YOU, within 1.5 s) | 5 | | | |
| where_are_you (WHERE, then YOU) | 5 | | | |
| where_are_you (YOU, then WHERE) | 5 | | | |
| not signing, 10 s (talk, move, scratch your face) | 3 | false triggers: | | |

Tips for counting: write "wrong word" together with what appeared (for example "you -> where"); a word that shows on the
small prediction line but is never committed counts as "nothing committed".

## 7. What to send back
1. The filled table.
2. The last lines of the 60-second latency run.
3. Lighting and distance ("window in front, 80 cm"), laptop model, Wi-Fi or hotspot.
4. Anything odd: a word that never appears, one that gets stuck on the screen, `SAFE MODE`, `reconnecting`, lag.
5. Optional: with **k** on, a note on whether the hand skeletons looked right during the words that failed.

Do not send screenshots of any dataset video, and never send or paste the token.

## If something goes wrong
| Symptom | Fix |
|---------|-----|
| `HTTP 401` | token differs: same value as the PC, new terminal |
| `reconnecting in Ns` | tunnel, PC server or Wi-Fi: run preflight check 5 again |
| Window is black or frozen | another app holds the camera; close it and restart the client |
| Word never appears | check keypoints with **k**; more light, both hands in view, hold the sign about 1 s |
| Words appear late | HOW, WHERE and YOU wait up to 1.5 s for a partner word; this is normal |
| Everything is slow | look at the round trip bottom right; Wi-Fi or internet, the server itself needs about 5 ms |
