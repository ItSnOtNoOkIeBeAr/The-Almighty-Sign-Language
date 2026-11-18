"""Export a trained SignNet to ONNX (fixed shape [1, 30, 233]), check it against PyTorch, build the TensorRT FP16 engine.

  python training/export_onnx.py --model rtmw_xl_384x288 --out server/models/temporal_primary
Writes <out>.onnx and <out>.engine (gitignored). ONNX is verified with onnxruntime-gpu against PyTorch on real windows
(max abs logit difference, argmax agreement); the engine is verified against ONNX Runtime through trtexec.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import ort_cuda
import signfeat as S
import sign_data as D
import train as T
from sign_model import SignNet

ROOT = Path(__file__).resolve().parent.parent
TRT = Path(r"C:\dev\TensorRT-10.16.1.11\bin\trtexec.exe")
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="path without extension")
    a = ap.parse_args()
    out = ROOT / a.out
    net = SignNet(n_cls=len(D.LABELS))
    net.load_state_dict(torch.load(T.RUNS / a.model / "best.pt", map_location="cpu"))
    net.eval()
    x = torch.zeros(1, S.WINDOW, S.F_DIM)
    torch.onnx.export(net, x, str(out.with_suffix(".onnx")), input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False)
    print("exported", out.with_suffix(".onnx"), f"{out.with_suffix('.onnx').stat().st_size / 1e6:.1f} MB")

    # real windows: val + test sets from the cached keypoints
    d = D.load(a.model)
    bank = D.Bank(d, "cuda")
    plan = T.eval_set(d, "test")
    pts, flg = bank.windows(plan["clip"], plan["t0"], plan["rate"], plan["lo"], plan["hi"])
    feats = S.window_features(pts, flg).cpu()
    with torch.no_grad():
        ref = net(feats).numpy()
    sess = ort_cuda.make_session(str(out.with_suffix(".onnx")), tf32=False)
    got = np.concatenate([sess.run(None, {"x": feats[i:i + 1].numpy()})[0] for i in range(len(feats))])
    diff = np.abs(ref - got).max()
    print(f"ONNX (ORT CUDA, exact FP32, batch 1) vs PyTorch on {len(feats)} real test windows: max |logit diff| {diff:.2e}, argmax agreement {(ref.argmax(1) == got.argmax(1)).mean():.4f}")
    assert diff < 1e-4, "ONNX output differs from PyTorch"
    tf = ort_cuda.make_session(str(out.with_suffix(".onnx")))
    got_tf = np.concatenate([tf.run(None, {"x": feats[i:i + 1].numpy()})[0] for i in range(len(feats))])
    print(f"  (info) ORT CUDA with its default TF32 matmuls: max |logit diff| {np.abs(ref - got_tf).max():.2e}, argmax agreement {(ref.argmax(1) == got_tf.argmax(1)).mean():.4f}")

    eng = out.with_suffix(".engine")
    env = {**__import__("os").environ, "PATH": f"{TRT.parent};{CUDA_BIN};" + __import__("os").environ["PATH"]}
    r = subprocess.run([str(TRT), f"--onnx={out.with_suffix('.onnx')}", "--fp16", f"--saveEngine={eng}"], capture_output=True, text=True, env=env)
    print("trtexec build exit", r.returncode, f"{eng.stat().st_size / 1e6:.1f} MB" if eng.exists() else r.stdout[-800:])
    assert r.returncode == 0

    # TensorRT vs ORT on 64 real windows (batch 1 engine, one trtexec run per window would be slow: use a 64-batch ONNX-free check via --loadInputs on window 0..15)
    work = out.parent / "_trt_check"
    work.mkdir(exist_ok=True)
    errs = []
    for i in range(0, len(feats), max(1, len(feats) // 16)):
        feats[i:i + 1].numpy().astype(np.float32).tofile(work / "w.bin")
        rr = subprocess.run([str(TRT), f"--loadEngine={eng}", "--loadInputs=x:w.bin", "--exportOutput=o.json", "--iterations=1", "--warmUp=0", "--duration=0"],
                            capture_output=True, text=True, env=env, cwd=work)
        vals = np.array(json.loads((work / "o.json").read_text())[0]["values"], np.float32)
        errs.append((np.abs(vals - got[i]).max(), int(vals.argmax() == got[i].argmax())))
    print(f"TensorRT FP16 vs ORT on {len(errs)} real windows: max |logit diff| {max(e[0] for e in errs):.3f}, argmax agreement {np.mean([e[1] for e in errs]):.2f}")
    lat = subprocess.run([str(TRT), f"--loadEngine={eng}", "--warmUp=500", "--duration=5", "--noDataTransfers"], capture_output=True, text=True, env=env).stdout
    print([l.split("] ")[-1] for l in lat.splitlines() if "GPU Compute Time" in l][:1])


if __name__ == "__main__":
    main()
