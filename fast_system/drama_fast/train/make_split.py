"""Create deterministic train/val/test splits for DRAMA-X fast supervision JSONL.

This script is intentionally simple and *deterministic*.

Key design: split by **clip id** (not by sample id) to avoid leakage across frames
from the same clip.

Run:
  python -m drama_fast.train.make_split \
    --input_jsonl /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_fast_sup_v3_topk5.jsonl \
    --out_dir ./splits_v3 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clip_key_from_sample_id(sample_id: str) -> str:
    """Group key for leakage-free split.

    Expected sample_id like: clip_1102_000372_frame_000372
    We group by everything before '_frame_'.
    """
    if "_frame_" in sample_id:
        return sample_id.split("_frame_")[0]
    # fallback: whole id
    return sample_id


def stable_hash_to_unit_interval(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    v = int(h[:8], 16)  # 32-bit
    return v / float(0xFFFFFFFF)


def read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def summarize(items: List[dict]) -> Dict[str, object]:
    n = len(items)
    risk_vals = []
    type_counter = Counter()
    for it in items:
        try:
            t0 = it.get("targets", [{}])[0]
            rv = float(t0.get("risk_score", 0.0))
            risk_vals.append(rv)
            tp = t0.get("type", "unknown")
            type_counter[tp] += 1
        except Exception:
            continue
    if risk_vals:
        risk_min = min(risk_vals)
        risk_max = max(risk_vals)
        risk_mean = sum(risk_vals) / len(risk_vals)
    else:
        risk_min = risk_max = risk_mean = None
    return {
        "num_samples": n,
        "risk_min": risk_min,
        "risk_mean": risk_mean,
        "risk_max": risk_max,
        "top_types": type_counter.most_common(20),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed_salt", type=str, default="drama-x")
    args = ap.parse_args()

    rsum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(rsum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {rsum}")

    _safe_mkdir(args.out_dir)
    items = read_jsonl(args.input_jsonl)
    print(f"Loaded {len(items)} samples from {args.input_jsonl}")

    # group by clip
    groups: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        sid = str(it.get("sample_id", ""))
        gk = clip_key_from_sample_id(sid)
        groups[gk].append(it)
    print(f"Grouped into {len(groups)} clips")

    train, val, test = [], [], []
    for gk, gitems in groups.items():
        u = stable_hash_to_unit_interval(args.seed_salt + "::" + gk)
        if u < args.train_ratio:
            train.extend(gitems)
        elif u < args.train_ratio + args.val_ratio:
            val.extend(gitems)
        else:
            test.extend(gitems)

    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    stats = {
        "input": os.path.abspath(args.input_jsonl),
        "out_dir": os.path.abspath(args.out_dir),
        "num_clips": len(groups),
        "split": {
            "train": summarize(train),
            "val": summarize(val),
            "test": summarize(test),
        },
    }
    stats_path = os.path.join(args.out_dir, "split_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Wrote:")
    print(f"  train: {train_path} ({len(train)})")
    print(f"  val  : {val_path} ({len(val)})")
    print(f"  test : {test_path} ({len(test)})")
    print(f"  stats: {stats_path}")


if __name__ == "__main__":
    main()
