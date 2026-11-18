"""Test-signer report (window level) for the DEPLOYED checkpoint of a model: per-class P/R/F1, confusion matrix, macro-F1.
  python training/report_test.py --model rtmw_xl_384x288        (uses training/runs/<model>/best.pt = the seed picked on val)
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import sign_data as D
import train as T
from sign_model import SignNet

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
d = D.load(a.model)
bank = D.Bank(d, "cuda")
net = SignNet(n_cls=len(D.LABELS)).cuda()
net.load_state_dict(torch.load(T.RUNS / a.model / "best.pt"))
for split in ("val", "test"):
    plan = T.eval_set(d, split)
    p = T.predict(net, bank, plan)
    y, yh = plan["label"], p.argmax(1)
    labs = list(range(len(D.LABELS)))
    print(f"\n=== {split.upper()} signers ({'used for early stopping and selection' if split == 'val' else 'never used for any choice'}), {a.model}, {len(y)} windows ===")
    print(classification_report(y, yh, labels=labs, target_names=D.LABELS, digits=3, zero_division=0))
    cm = confusion_matrix(y, yh, labels=labs)
    print("confusion matrix (rows = true, cols = predicted):\n" + " " * 8 + "".join(f"{n:>7}" for n in D.LABELS))
    for i, row in enumerate(cm):
        print(f"{D.LABELS[i]:>8}" + "".join(f"{v:7d}" for v in row))
    rep = classification_report(y, yh, labels=labs, target_names=D.LABELS, output_dict=True, zero_division=0)
    if split == "test":
        json.dump({"macro_f1": rep["macro avg"]["f1-score"], "accuracy": rep["accuracy"], "report": rep, "confusion": cm.tolist()}, open(T.RUNS / a.model / "test_report.json", "w"), indent=1)
