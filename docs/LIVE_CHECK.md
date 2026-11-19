# Live check from the laptop (through the Cloudflare tunnel)

Purpose: see how the trained model behaves on a real webcam. Nothing is recorded or saved; this is a check, not training data.

## Before you start (on the PC)
1. A server running with the real model: `powershell -File server\scripts\switch_server.ps1 -To cpp` (C++ server, the normal setup;
   `-To python` for the Python fallback). `https://signlang.dheynorlanda.me/health` must answer `{"ok":true,"server":"cpp",...}`.
2. The `cloudflared` service is Running (`sc query cloudflared`).

## On the laptop
```
cd <repo>
git fetch origin
git reset --hard origin/main      # not "git pull": the repository history was rewritten on 2026-11-19
client\.venv\Scripts\python client\client.py
```
Keys: `k` shows the detected hand and body keypoints on the video (handy to see why a sign is missed), `d` shows more details.
(The config already points to `wss://signlang.dheynorlanda.me/ws`; 640x480, 10 fps, JPEG q70. `SIGNLANG_TOKEN` must be set.
Press `q` or Esc to quit.) The overlay shows the live prediction, its confidence, the state (idle / signing / committed),
the **committed** words (a sentence appears as `how_are_you` / `where_are_you`) and the round-trip time.

## Setup that matters
- Bright, even light on your face and hands (a window or lamp in front of you, not behind). The model was trained on
  well-lit webcam video; a dark image gives no hands.
- Sit about 0.7-1 m from the camera so your face, shoulders and both hands fit in the frame. Plain background, nobody else visible.
- Start and end every try with your hands down, out of the frame (below the shoulders). The system only wakes up when it sees hands.
- ASL signs, one at a time, as in the ASL Citizen videos (isolated signs, held clearly for about one second).

## The test: 10 tries per word
For each of HELLO, WHAT, HOW, WHY, WHERE, YOU: hands down, sign once, hands down, wait for the committed word (up to 2 s for
HOW / WHERE / YOU, which wait for a partner word), note what was committed, repeat 10 times.
Then 5 tries each of the sentences: HOW then YOU, WHERE then YOU, YOU then WHERE (sign them one right after the other,
within about 1.5 s), and 10 seconds of NOT signing (talk, move, scratch your face) to count false triggers.

| Word | Tries | Correct | Wrong word | Nothing committed |
|------|-------|---------|------------|-------------------|
| hello | 10 | | | |
| what | 10 | | | |
| how | 10 | | | |
| why | 10 | | | |
| where | 10 | | | |
| you | 10 | | | |
| how_are_you (HOW, YOU) | 5 | | | |
| where_are_you (WHERE, YOU) | 5 | | | |
| where_are_you (YOU, WHERE) | 5 | | | |
| not signing, 10 s | 3 | false triggers: | | |

Send me the filled table and say which words went wrong, plus the lighting and distance you used.
