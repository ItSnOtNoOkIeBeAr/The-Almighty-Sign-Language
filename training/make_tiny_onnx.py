"""Builds a tiny conv+relu+matmul ONNX model, used only to smoke-test the toolchain
(trtexec FP16 engine build, onnxruntime CUDAExecutionProvider). Not a real model."""
import sys

import torch
import torch.nn as nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x)).mean(dim=(2, 3))
        return self.fc(x)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny.onnx"
    torch.manual_seed(0)
    # dynamo=False: the legacy exporter writes a single self-contained file.
    torch.onnx.export(Tiny().eval(), torch.randn(1, 3, 64, 64), out,
                      input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    print("wrote", out)
