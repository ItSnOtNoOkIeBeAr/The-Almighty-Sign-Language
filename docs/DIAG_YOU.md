# Live diagnostic: does the model ever give YOU any probability?

Purpose: when a word never appears live, see what the model actually predicts, frame by frame. The server writes one CSV line per
frame to `logs\diag_<time>.csv`: the probability of every class, the top 3, the decision state, and how many hand keypoints were
detected. Numbers only: no images, no video, nothing leaves the PC. `logs\` is gitignored.

## On the PC (the C++ server must be restarted once to turn the log on)
1. Stop the running server (Ctrl+C in its terminal, or `powershell -File server\scripts\switch_server.ps1 -To stop`).
2. Rebuild once so the exe knows `--diag-log`: `powershell -File server\scripts\build_cpp.ps1`
3. Start it with the log on, in the foreground like before:
   `powershell -File server\scripts\start_server.ps1 -Foreground -Diag`
   (it prints `diagnostic log: C:\dev\almighty-sign-language\logs\diag_<time>.csv`; background alternative: `switch_server.ps1 -To cpp -Diag`).
   Python fallback: `switch_server.ps1 -To python -Diag`.

## On the laptop (10 YOU tries, about 2 minutes)
1. `client\.venv\Scripts\python client\client.py --width 640 --height 480`, press **k** (keypoints).
2. Hands down, point at the camera as you would for YOU (one second), hands down. Repeat 10 times with a 2-second pause.
   Do it the way you tried live before, not "better".
3. Quit with `q`.

## On the PC: read the log
```
client\.venv\Scripts\python server\scripts\diag_report.py logs\diag_<time>.csv --word you
```
It prints: share of frames with hands up, hand keypoints detected (a hand needs 15 of 21), per class the highest probability and
how many frames passed 0.1 / 0.3 / 0.5, and one line per hands-up burst (= one try) with the peak p(you).

How to read it:
| Result | Meaning |
|---|---|
| `hand pts` far below 15 while pointing | the hand is not detected when it points at the camera: the model gets no hand, a keypoint problem |
| hands detected, peak p(you) below 0.1 in every burst | the model does not recognise your YOU: a data / style problem |
| peak p(you) 0.3-0.5 but never committed | the decision threshold or the number of frames (M) is too strict for a short sign |
| p(you) high but another word wins | confusion with that word (see the "best non-none" column) |

Send me the whole report output (it contains no personal data).

## Replay a log through the decision logic (double commits, threshold questions)
`training\.venv\Scripts\python server\scripts\replay_diag.py logs\diag_<time>.csv --list [--set repeat_rest_frames=3 --set repeat_cooldown_ms=2000]`
replays the logged probabilities and hand flags through `decision.py` (first it checks that the replay equals what the server committed),
lists every commit, flags the same word committed twice within 6 s, and tells whether p(word) dipped in between (two signs) or not (one sign).
