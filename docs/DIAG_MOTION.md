# Motion-gate recording (about 8 minutes)

Goal: one diagnostic run with a clear "not signing" part and a clear "words" part, with phase markers made by you (no guessing),
so the motion gate's threshold can be chosen on real data. The log holds numbers only (probabilities, hand flags, hand motion): no images.

## 0. Before: the PC server must be the NEW build
On the PC (rebuilds the release exe, so stop the running server first):
```
powershell -File server\scripts\switch_server.ps1 -To stop
powershell -File server\scripts\build_cpp.ps1
powershell -File server\scripts\start_server.ps1 -Foreground -Diag
```
`-Diag` writes `logs\diag_<time>.csv` (it now has the columns `mot_*`).

## 1. Run (client machine: this PC or the laptop)
Terminal A: the client window, as usual (`--width 640 --height 480` on this PC).
```
client\.venv\Scripts\python client\client.py --width 640 --height 480
```
Terminal B (same machine as the client, repo root): the markers. Run each line at the moment the phase starts:
```
client\.venv\Scripts\python client\mark.py no_sign     # start of the not-signing part
client\.venv\Scripts\python client\mark.py gap         # after about 3 minutes
client\.venv\Scripts\python client\mark.py words       # after the 20 s calm gap
client\.venv\Scripts\python client\mark.py end         # when you are done
```
| Phase | What to do | Length |
|---|---|---|
| `no_sign` | typing, resting hands on the desk, fixing hair, talking, scratching your face, drinking, waving; NO signing | about 3 min |
| `gap` | hands down, out of frame, still, quiet | 20 s |
| `words` | each word 5 times, hands down between tries: hello, what, how, why, where, you (sentences are optional) | until done |

Start `mark.py no_sign` after the window says `connected`. Press `q` in the client and stop the server after `mark.py end`.

## 2. Mark the phases
On the PC, in the repo root (if the client ran on the laptop, copy the laptop's `logs\markers.csv` to the PC's `logs\` folder first; it holds only names and times):
```
client\.venv\Scripts\python server\scripts\mark_phases.py logs\diag_<time>.csv
```
It prints, per phase, the length, ticks, share of ticks with a hand in view, mean hand motion and the committed words, and warns if the "calm" gap was not calm.
Without `mark.py` use your clock instead: `--at no_sign=21:04:10 --at gap=21:07:10 --at words=21:07:30 --at end=21:13:00` (local time of the client machine).

## 3. Send back
The table printed by `mark_phases.py`. Then (assistant side) `training\guard_experiments.py --live logs\diag_<time>.csv --phases logs\diag_<time>.phases.json` chooses the threshold.
