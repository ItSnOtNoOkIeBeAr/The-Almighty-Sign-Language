"""Selective extraction from ASL_Citizen.zip (never extract the whole archive).

  python training/extract_asl_citizen.py --zip C:\\dev\\Datasets\\ASL_Citizen.zip --out data/public/asl_citizen

Extracts every clip of the target glosses plus ALL clips of a seeded random sample of other glosses (class
"other"), keeps the official train/val/test split, checks every file's CRC32 against the zip directory, and
writes manifest.csv. The data is licensed for non-commercial research and must stay local (gitignored).
"""
import argparse
import csv
import io
import random
import re
import zipfile
import zlib
from pathlib import Path

# gloss -> class. HI has no gloss of its own in ASL Citizen (see governance/DECISIONS.md).
TARGETS = {"HELLO": "hello", "HOW1": "how", "HOW2": "how", "WHAT1": "what", "WHAT2": "what",
           "WHERE": "where", "WHY": "why", "YOU": "you"}
# OTHER must not contain signs that are (nearly) our words or built from them.
NEAR_TARGET = re.compile(r"HELLO|HOW|WHAT|WHY|WHERE|YOU|WAVE|GREET", re.I)


def read_rows(z):
    rows = []
    for split in ("train", "val", "test"):
        text = z.read(f"ASL_Citizen/splits/{split}.csv").decode("utf-8-sig")
        for r in csv.DictReader(io.StringIO(text)):
            rows.append({"split": split, "signer": r["Participant ID"], "file": r["Video file"], "gloss": r["Gloss"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--other-glosses", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(a.zip) as z:
        rows = read_rows(z)
        pool = sorted({r["gloss"] for r in rows if r["gloss"] not in TARGETS and not NEAR_TARGET.search(r["gloss"])})
        other = set(random.Random(a.seed).sample(pool, a.other_glosses))
        chosen = [dict(r, cls=TARGETS.get(r["gloss"], "other")) for r in rows if r["gloss"] in TARGETS or r["gloss"] in other]
        print(f"{len(rows)} clips in csv; extracting {len(chosen)} ({len(other)} other glosses, seed {a.seed})", flush=True)
        total = 0
        for n, r in enumerate(chosen, 1):
            info = z.getinfo(f"ASL_Citizen/videos/{r['file']}")
            data = z.read(info)  # also raises BadZipFile on a CRC mismatch
            if zlib.crc32(data) != info.CRC:
                raise RuntimeError(f"CRC mismatch: {r['file']}")
            (out / "videos" / r["file"]).write_bytes(data)
            r.update(crc32=f"{info.CRC:08x}", bytes=info.file_size)
            total += info.file_size
            if n % 500 == 0:
                print(f"  {n}/{len(chosen)}  {total / 1e9:.2f} GB", flush=True)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["split", "signer", "file", "gloss", "cls", "crc32", "bytes"])
        w.writeheader()
        w.writerows(chosen)
    print(f"done: {len(chosen)} files, {total / 1e9:.2f} GB, all CRC OK")


if __name__ == "__main__":
    main()
