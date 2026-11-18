"""Creates the PLACEHOLDER temporal model: a correctly shaped GRU classifier with RANDOM weights.

PLACEHOLDER: it recognises nothing. It exists so the whole pipeline runs before any data exists.
The trained SignNet (train.py / export_onnx.py) has the same I/O contract: x [1, 30, 233] float32 -> logits [1, 7];
this file stays only as a smoke-test fixture for the servers.
Usage: python make_placeholder_model.py [out.onnx] [frames] [feature_dim]
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
NUM_CLASSES = len(json.loads((ROOT / "protocol" / "labels.json").read_text())["labels"])


class PlaceholderTemporal(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "server" / "models" / "temporal_placeholder.onnx")
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    feat = int(sys.argv[3]) if len(sys.argv) > 3 else 233
    torch.manual_seed(0)
    # dynamo=False: the legacy exporter writes one self-contained file, which TensorRT's parser also likes.
    torch.onnx.export(PlaceholderTemporal(feat).eval(), torch.zeros(1, frames, feat), out,
                      input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False)
    print(f"PLACEHOLDER temporal model written: {out}  input [1,{frames},{feat}] -> [1,{NUM_CLASSES}]")
