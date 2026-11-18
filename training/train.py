"""Phase 7 training: SignNet on ASL Citizen keypoints, official signer split.

  python training/train.py --model rtmw_xl_384x288 --seeds 3     # primary keypoints
  python training/train.py --model rtmw_lm_256x192 --seeds 3     # fallback keypoints
Train on `train`, early-stop on `val` macro-F1, report on `test` signers only (never used for any choice).
Everything stays on the GPU; bf16 autocast; augmentation in sample_epoch/augment. Outputs go to training/runs/<model>/ (gitignored).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

import signfeat as S
import sign_data as D
from sign_model import SignNet

RUNS = Path(__file__).resolve().parent / "runs"
REPEATS = 24          # augmented windows per word clip per epoch
COMPOUND = 0.35       # share of word windows that start with an earlier sign (sentences / fast signing)
NEW_SIGN_NONE = 0.0   # tried 0.15: `none` windows with an earlier sign + the first frames of a new one; it raised false triggers and broke sentence joining (DECISIONS.md)
NONE_RATIO = 3        # per-epoch `none` windows = at most this x the largest word class
EVAL_SHIFTS = (-3, 0, 3)


def active_t0(a0, a1, rate, rng, shift=None):
    """Window start for a clip whose sign spans frames a0..a1. shift=None: random placement (training); else centred +- shift frames."""
    length, span = a1 - a0 + 1, (S.WINDOW - 1) * rate
    if length > 0.9 * span:  # sign longer than the window: any window inside the sign
        return rng.uniform(a0, max(a0, a1 - span)) if shift is None else (a0 + a1) / 2 - span / 2 + shift * rate
    m = max(0.0, (span - length) / 2 + 0.2 * length)  # keeps at least ~70% of the sign inside the window
    return (a0 + a1) / 2 - span / 2 + (rng.uniform(-m, m) if shift is None else shift * rate)


def rest_segments(d, split):
    """(clip, lo, hi) frame ranges of resting pose before/after the sign (>= 2 frames, 1 frame margin to the sign)."""
    out = []
    for c in np.where(d["split"] == split)[0]:
        a0, a1 = d["seg"][c]
        n = d["len"][c]
        if a0 - 2 >= 1:
            out.append((c, 0, a0 - 2))
        if n - 1 - (a1 + 2) >= 1:
            out.append((c, a1 + 2, n - 1))
    return out


def sample_epoch(d, rng):
    """Training windows for one epoch -> dict of arrays. Words x REPEATS, none <= NONE_RATIO x largest word class."""
    tr = d["split"] == "train"
    plan = {k: [] for k in ("clip", "label", "t0", "rate", "lo", "hi", "clip2", "t02", "k")}

    def add(c, label, t0, rate, lo=None, hi=None, clip2=-1, t02=0.0, k=0):
        plan["clip"].append(c), plan["label"].append(label), plan["t0"].append(t0), plan["rate"].append(rate)
        plan["lo"].append(0 if lo is None else lo), plan["hi"].append(d["len"][c] - 1 if hi is None else hi)
        plan["clip2"].append(clip2), plan["t02"].append(t02), plan["k"].append(k)

    word = np.where(tr & (d["cls"] > 0))[0]
    active = np.where(tr)[0]  # earlier signs for compound windows: any word or other-gloss clip
    counts = np.zeros(len(D.LABELS), int)
    for c in word:
        for _ in range(REPEATS):
            r = rng.uniform(0.8, 1.2)
            if rng.random() < COMPOUND:
                # earlier sign A ends (gap 1-6 frames) before the target sign starts at window frame k; the label is the later sign
                k, gap, a = int(rng.integers(8, 19)), int(rng.integers(1, 7)), rng.choice(active)
                add(c, d["cls"][c], d["seg"][c][0] - k * r + rng.uniform(-1.5, 1.5) * r, r,
                    clip2=a, t02=d["seg"][a][1] - (k - gap - 1) * r, k=k)
            else:
                add(c, d["cls"][c], active_t0(*d["seg"][c], r, rng), r)
            counts[d["cls"][c]] += 1
    n_none = NONE_RATIO * counts.max()
    other = np.where(tr & (d["cls"] == 0))[0]
    rest = rest_segments(d, "train")
    for i in range(n_none):
        r = rng.uniform(0.8, 1.2)
        if NEW_SIGN_NONE and rng.random() < NEW_SIGN_NONE:
            k, gap, a, c = int(rng.integers(22, 28)), int(rng.integers(1, 7)), rng.choice(active), rng.choice(active)
            add(c, 0, d["seg"][c][0] - k * r, r, clip2=a, t02=d["seg"][a][1] - (k - gap - 1) * r, k=k)
        elif i % 2 == 0 or not rest:
            c = rng.choice(other)
            add(c, 0, active_t0(*d["seg"][c], r, rng), r)
        else:
            c, lo, hi = rest[rng.integers(len(rest))]
            add(c, 0, rng.uniform(lo - 8, hi), r, lo, hi)
    counts[0] = n_none
    return {k: np.array(v) for k, v in plan.items()}, counts


def eval_set(d, split, seed=0):
    """Deterministic windows: every word clip at 3 shifts; `none` = other-gloss clips + resting windows, each capped at NONE_RATIO x largest word class."""
    rng = np.random.default_rng(seed)
    plan = {k: [] for k in ("clip", "label", "t0", "rate", "lo", "hi", "clip2", "t02", "k")}

    def add(c, label, t0, lo=None, hi=None):
        plan["clip"].append(c), plan["label"].append(label), plan["t0"].append(t0), plan["rate"].append(1.0)
        plan["lo"].append(0 if lo is None else lo), plan["hi"].append(d["len"][c] - 1 if hi is None else hi)
        plan["clip2"].append(-1), plan["t02"].append(0.0), plan["k"].append(0)

    sp = d["split"] == split
    cnt = np.zeros(len(D.LABELS), int)
    for c in np.where(sp & (d["cls"] > 0))[0]:
        for s in EVAL_SHIFTS:
            add(c, d["cls"][c], active_t0(*d["seg"][c], 1.0, rng, s))
            cnt[d["cls"][c]] += 1
    cap = NONE_RATIO * cnt.max()
    other = np.where(sp & (d["cls"] == 0))[0]
    for c in rng.choice(other, min(cap // 2, len(other)), replace=False):
        add(c, 0, active_t0(*d["seg"][c], 1.0, rng, 0))
    rest = rest_segments(d, split)
    for i in rng.choice(len(rest), min(cap // 2, len(rest)), replace=False):
        c, lo, hi = rest[i]
        add(c, 0, (lo + hi) / 2 - (S.WINDOW - 1) / 2, lo, hi)
    return {k: np.array(v) for k, v in plan.items()}


def augment(norm, flg, g):
    """GPU augmentation on normalised windows [B,T,57,2] / flags [B,T,3]: rotation, scale, shift, noise, frame drop, missing hands."""
    B, T = norm.shape[:2]
    dev = norm.device
    ang = (torch.rand(B, device=dev, generator=g) * 2 - 1) * math.radians(10)
    cs, sn = ang.cos(), ang.sin()
    rot = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)          # [B,2,2]
    scale = 0.9 + 0.2 * torch.rand(B, 1, 1, device=dev, generator=g)
    shift = (torch.rand(B, 1, 1, 2, device=dev, generator=g) * 2 - 1) * 0.1
    norm = (norm @ rot[:, None].transpose(-1, -2)) * scale[..., None] + shift
    norm = norm + torch.randn(norm.shape, device=dev, generator=g) * 0.01
    # frame drop: a dropped frame repeats the previous one (the live stream also stalls)
    drop = torch.rand(B, T, device=dev, generator=g) < 0.1
    drop[:, 0] = False
    src = torch.arange(T, device=dev)[None].expand(B, T).clone()
    for t in range(1, T):
        src[:, t] = torch.where(drop[:, t], src[:, t - 1], src[:, t])
    norm = torch.gather(norm, 1, src[:, :, None, None].expand_as(norm))
    flg = torch.gather(flg, 1, src[:, :, None].expand_as(flg))
    # missing hand: a contiguous span of 20-60% of the frames
    for h in (0, 1):
        hit = torch.rand(B, device=dev, generator=g) < 0.12
        ln = (T * (0.2 + 0.4 * torch.rand(B, device=dev, generator=g))).long()
        st = (torch.rand(B, device=dev, generator=g) * (T - ln)).long()
        t = torch.arange(T, device=dev)[None]
        span = hit[:, None] & (t >= st[:, None]) & (t < (st + ln)[:, None])
        flg = flg.clone()
        flg[..., h] = flg[..., h] & ~span
        flg[..., 3 + h] = flg[..., 3 + h] & ~span
    keep = torch.ones(B, T, S.N_PTS, dtype=torch.bool, device=dev)
    keep[:, :, S.L0:S.R0] = flg[..., 0:1]
    keep[:, :, S.R0:] = flg[..., 1:2]
    return norm * keep[..., None], flg


def batches(plan, n, bs, shuffle, rng=None):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


@torch.no_grad()
def predict(model, bank, plan):
    model.eval()
    out = []
    for idx in batches(plan, len(plan["clip"]), 256, False):
        pts, flg = bank.windows2(plan["clip"][idx], plan["t0"][idx], plan["rate"][idx], plan["lo"][idx], plan["hi"][idx], plan["clip2"][idx], plan["t02"][idx], plan["k"][idx])
        out.append(model(S.window_features(pts, flg)).float().softmax(-1).cpu())
    return torch.cat(out).numpy()


def metrics(probs, labels):
    pred = probs.argmax(1)
    return {"macro_f1": float(f1_score(labels, pred, average="macro", labels=list(range(len(D.LABELS))), zero_division=0)),
            "acc": float((pred == labels).mean())}


def train_one(d, bank, seed, epochs, val_plan, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    model = SignNet(n_cls=len(D.LABELS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = epochs * 40
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    best, best_state, bad, hist = (-1.0, 1e9), None, 0, []
    for ep in range(epochs):
        plan, counts = sample_epoch(d, rng)
        w = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)  # inverse-frequency class weights
        lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
        model.train()
        n, tl, nb = len(plan["clip"]), 0.0, 0
        for idx in batches(plan, n, 128, True, rng):
            if nb >= 40:
                break
            pts, flg = bank.windows2(plan["clip"][idx], plan["t0"][idx], plan["rate"][idx], plan["lo"][idx], plan["hi"][idx], plan["clip2"][idx], plan["t02"][idx], plan["k"][idx])
            norm, flg = augment(S.normalize_window(pts, flg), flg, g)
            y = torch.from_numpy(plan["label"][idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(S.to_features(norm, flg)).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step() if sched.last_epoch < steps - 1 else None
            tl, nb = tl + loss.item(), nb + 1
        pv = predict(model, bank, val_plan)
        m = metrics(pv, val_plan["label"])
        vloss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(pv) + 1e-8), torch.from_numpy(val_plan["label"])))
        hist.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vloss, **m})
        key = (m["macro_f1"], -vloss)
        if key > (best[0], -best[1]):
            best, bad = (m["macro_f1"], vloss), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f"  seed {seed} ep {ep:3d} train {tl / max(nb, 1):.3f} val loss {vloss:.3f} F1 {m['macro_f1']:.3f} acc {m['acc']:.3f}{' *' if bad == 0 else ''}", flush=True)
        if bad >= 15:
            break
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_val_macro_f1": best[0], "best_val_loss": best[1], "epochs_run": len(hist), "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rtmw_xl_384x288", "rtmw_lm_256x192"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    dev = "cuda"
    d = D.load(a.model)
    print(f"{a.model}: {len(d['len'])} clips, {len(d['pts'])} frames; active segment found in {d['seg_found'].mean():.1%} of clips; "
          f"shoulders reliable in {d['flg'][:, 2].mean():.1%} of frames, left hand {d['flg'][:, 0].mean():.1%}, right hand {d['flg'][:, 1].mean():.1%}", flush=True)
    bank = D.Bank(d, dev)
    val_plan, test_plan = eval_set(d, "val"), eval_set(d, "test")
    print(f"val windows {len(val_plan['clip'])} (none {int((val_plan['label'] == 0).sum())}), test windows {len(test_plan['clip'])} (none {int((test_plan['label'] == 0).sum())})", flush=True)
    out = RUNS / a.model
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(a.seeds):
        t0 = time.time()
        model, info = train_one(d, bank, seed, a.epochs, val_plan, dev)
        info["seconds"] = round(time.time() - t0)
        torch.save(model.state_dict(), out / f"seed{seed}.pt")
        runs.append(info)
        print(f"seed {seed}: best val macro-F1 {info['best_val_macro_f1']:.3f} (loss {info['best_val_loss']:.3f}), {info['epochs_run']} epochs, {info['seconds']} s", flush=True)
    best = max(runs, key=lambda r: (r["best_val_macro_f1"], -r["best_val_loss"]))
    model = SignNet(n_cls=len(D.LABELS)).to(dev)
    model.load_state_dict(torch.load(out / f"seed{best['seed']}.pt"))
    torch.save(model.state_dict(), out / "best.pt")
    pt = predict(model, bank, test_plan)
    y = test_plan["label"]
    names = D.LABELS
    print(f"\n=== TEST signers, {a.model}, best seed {best['seed']} (chosen on val) ===")
    print(classification_report(y, pt.argmax(1), labels=list(range(len(names))), target_names=names, digits=3, zero_division=0))
    cm = confusion_matrix(y, pt.argmax(1), labels=list(range(len(names))))
    print("confusion matrix (rows = true, cols = predicted):\n" + " " * 8 + "".join(f"{n:>7}" for n in names))
    for i, row in enumerate(cm):
        print(f"{names[i]:>8}" + "".join(f"{v:7d}" for v in row))
    res = {"model": a.model, "params": sum(p.numel() for p in model.parameters()), "val_macro_f1_per_seed": [r["best_val_macro_f1"] for r in runs],
           "val_macro_f1_mean": float(np.mean([r["best_val_macro_f1"] for r in runs])), "best_seed": best["seed"],
           "test": {**metrics(pt, y), "report": classification_report(y, pt.argmax(1), labels=list(range(len(names))), target_names=names, output_dict=True, zero_division=0),
                    "confusion": cm.tolist()},
           "runs": runs}
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print(f"params {res['params']:,}  val macro-F1 per seed {res['val_macro_f1_per_seed']}  mean {res['val_macro_f1_mean']:.3f}  TEST macro-F1 {res['test']['macro_f1']:.3f} acc {res['test']['acc']:.3f}")


if __name__ == "__main__":
    main()
