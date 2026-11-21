r"""Write a phase marker for a diagnostic run (standard library only; use it on the machine that runs the client, in a second terminal).

  client\.venv\Scripts\python client\mark.py no_sign     # when you start the "not signing" part
  client\.venv\Scripts\python client\mark.py gap         # when the calm gap starts
  client\.venv\Scripts\python client\mark.py words       # when the first word starts
  client\.venv\Scripts\python client\mark.py end         # when you are done
Appends "name,epoch_ms" to logs\markers.csv (same clock as the frame timestamps of this machine). server\scripts\mark_phases.py turns the markers
and a diag log into phases. Numbers only.
"""
import sys
import time
from pathlib import Path

NAMES = ("no_sign", "gap", "words", "end")

if len(sys.argv) != 2 or sys.argv[1] not in NAMES:
    sys.exit("usage: mark.py " + "|".join(NAMES))
path = Path(__file__).resolve().parent.parent / "logs" / "markers.csv"
path.parent.mkdir(exist_ok=True)
now = time.time()
with open(path, "a", encoding="utf-8", newline="\n") as f:
    f.write(f"{sys.argv[1]},{int(now * 1000)}\n")
print(f"marker {sys.argv[1]} at {time.strftime('%H:%M:%S', time.localtime(now))} -> {path}")
