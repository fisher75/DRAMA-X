#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA gate for DRAMA-X CoC jsonl (rule-based v2 full).
- risk margin (top1-top2 risk_score)
- trace info quality heuristics
- decision consistency checks
- export review samples
Only uses Python stdlib.

功能：
统计：总量、risk_label 分布、driving_decision 分布
risk margin：top1-top2 的 score margin 分布（并输出低 margin 列表）
decision consistency：risk_label vs decision 的冲突统计
trace 信息量：长度、是否包含关键字段（type/intent/position 等）、模板化检测
导出 人工抽检样本：随机 + 低 margin + 低信息 trace（可控数量）
输出：report.md、stats.json、samples_for_review.jsonl、low_margin.jsonl、low_info_trace.jsonl
"""
"""
python qa_gate_drama_coc.py \
  --input /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b_v2_full_rule_based_20260108.jsonl \
  --out_dir /workspace/chz/code/DRAMA-X/annotation_coc/_qa_v2_rule \
  --low_margin_thr 0.10 \
  --sample_random 80 \
  --sample_low_margin 80 \
  --sample_low_info 80
"""


import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as e:
                raise RuntimeError(f"[JSON parse fail] {path}:{ln} -> {e}\nLINE={line[:2000]}")


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_sample_id(r: Dict[str, Any]) -> str:
    return str(safe_get(r, ["sample_id", "id", "clip_id", "uid", "name"], "UNKNOWN"))


def get_risk_label(r: Dict[str, Any]) -> str:
    x = safe_get(r, ["risk_label", "risk", "riskLevel", "risk_level"], "unknown")
    return str(x).lower()


def get_decision(r: Dict[str, Any]) -> str:
    x = safe_get(r, ["driving_decision", "decision", "action"], "unknown")
    return str(x).upper()


def get_trace(r: Dict[str, Any]) -> str:
    x = safe_get(r, ["coc_trace", "trace", "reasoning", "explanation"], "")
    return "" if x is None else str(x)


def get_primary_vru(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # try common placements
    cc = r.get("critical_components") or {}
    pv = None
    if isinstance(cc, dict):
        pv = cc.get("primary_vru") or cc.get("primary") or None
    if pv is None:
        pv = r.get("primary_vru") or r.get("primary") or None
    return pv if isinstance(pv, dict) else None


def get_vru_list(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = r.get("vru_list") or r.get("vrus") or r.get("objects") or []
    return v if isinstance(v, list) else []


def get_vru_score(vru: Dict[str, Any]) -> Optional[float]:
    s = safe_get(vru, ["risk_score", "score", "riskScore", "risk"], None)
    if s is None:
        return None
    try:
        return float(s)
    except:
        return None


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def trace_info_flags(trace: str, primary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    t = normalize_text(trace).lower()
    flags = {}
    flags["len"] = len(t)
    flags["has_because"] = ("because" in t) or ("由于" in t) or ("因为" in t)
    flags["has_therefore"] = ("therefore" in t) or ("所以" in t) or ("因此" in t)
    # very rough "templatey" check
    flags["template_like"] = bool(re.search(r"because .* therefore .*", t))
    # keyword coverage
    pv_type = ""
    pv_intent = ""
    pv_pos = ""
    if primary:
        pv_type = str(safe_get(primary, ["type", "category", "cls"], "")).lower()
        pv_intent = str(safe_get(primary, ["intent", "motion", "behavior"], "")).lower()
        pv_pos = str(safe_get(primary, ["position", "loc", "region"], "")).lower()
    flags["mentions_type"] = (pv_type and pv_type in t)
    flags["mentions_intent"] = (pv_intent and pv_intent in t)
    flags["mentions_position"] = (pv_pos and pv_pos in t)

    # low-info heuristic: too short OR no causal markers OR no mention of key facts
    flags["low_info"] = (flags["len"] < 80) or (not flags["has_because"] and not flags["has_therefore"]) or (
        (primary is not None) and (not (flags["mentions_type"] or flags["mentions_intent"] or flags["mentions_position"]))
    )
    return flags


def decision_consistency(risk_label: str, decision: str) -> str:
    # heuristic mapping; tune later if you want
    risk_label = risk_label.lower()
    decision = decision.upper()
    if risk_label in ("high", "danger", "hazard"):
        if decision in ("GO",):
            return "conflict_high_go"
    if risk_label in ("low", "safe"):
        if decision in ("STOP",):
            return "conflict_low_stop"
    return "ok"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CoC full jsonl (rule-based v2 full)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low_margin_thr", type=float, default=0.10, help="threshold for top1-top2 score margin")
    ap.add_argument("--sample_random", type=int, default=80)
    ap.add_argument("--sample_low_margin", type=int, default=80)
    ap.add_argument("--sample_low_info", type=int, default=80)
    args = ap.parse_args()

    random.seed(args.seed)
    ensure_dir(args.out_dir)

    total = 0
    risk_counter = Counter()
    decision_counter = Counter()
    consistency_counter = Counter()
    trace_len_list = []
    low_margin_records = []
    low_info_records = []
    all_records_cache = []  # keep minimal fields for sampling

    # margin hist bins
    margin_bins = Counter()

    for ln, r in read_jsonl(args.input):
        total += 1
        sid = get_sample_id(r)
        risk = get_risk_label(r)
        dec = get_decision(r)
        tr = get_trace(r)
        pv = get_primary_vru(r)
        vru_list = get_vru_list(r)

        risk_counter[risk] += 1
        decision_counter[dec] += 1

        cons = decision_consistency(risk, dec)
        consistency_counter[cons] += 1

        flags = trace_info_flags(tr, pv)
        trace_len_list.append(flags["len"])

        # margin
        s1, s2 = None, None
        if len(vru_list) >= 1:
            s1 = get_vru_score(vru_list[0])
        if len(vru_list) >= 2:
            s2 = get_vru_score(vru_list[1])
        margin = None
        if (s1 is not None) and (s2 is not None):
            margin = float(s1 - s2)
            # bin
            b = round(margin, 2)
            margin_bins[b] += 1
            if margin < args.low_margin_thr:
                low_margin_records.append({
                    "line": ln,
                    "sample_id": sid,
                    "margin": margin,
                    "top1_score": s1,
                    "top2_score": s2,
                    "risk_label": risk,
                    "decision": dec,
                    "primary_vru": pv,
                    "coc_trace": tr,
                })

        if flags["low_info"]:
            low_info_records.append({
                "line": ln,
                "sample_id": sid,
                "risk_label": risk,
                "decision": dec,
                "primary_vru": pv,
                "trace_len": flags["len"],
                "flags": {k: v for k, v in flags.items() if k != "len"},
                "coc_trace": tr,
            })

        all_records_cache.append({
            "line": ln,
            "sample_id": sid,
            "risk_label": risk,
            "decision": dec,
            "primary_vru": pv,
            "coc_trace": tr,
        })

    # sampling for review
    def sample_list(lst, k):
        if k <= 0:
            return []
        if len(lst) <= k:
            return lst
        return random.sample(lst, k)

    samples = []
    samples += sample_list(all_records_cache, args.sample_random)
    samples += sample_list(low_margin_records, args.sample_low_margin)
    samples += sample_list(low_info_records, args.sample_low_info)

    # de-dup by (line)
    seen = set()
    dedup = []
    for s in samples:
        key = s.get("line")
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)
    samples = dedup

    # write outputs
    stats = {
        "total": total,
        "risk_label_dist": dict(risk_counter),
        "decision_dist": dict(decision_counter),
        "consistency_dist": dict(consistency_counter),
        "low_margin_thr": args.low_margin_thr,
        "low_margin_count": len(low_margin_records),
        "low_info_count": len(low_info_records),
        "trace_len_min": min(trace_len_list) if trace_len_list else None,
        "trace_len_p50": sorted(trace_len_list)[len(trace_len_list)//2] if trace_len_list else None,
        "trace_len_p90": sorted(trace_len_list)[int(len(trace_len_list)*0.9)] if trace_len_list else None,
        "trace_len_max": max(trace_len_list) if trace_len_list else None,
    }

    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(args.out_dir, "samples_for_review.jsonl"), samples)
    write_jsonl(os.path.join(args.out_dir, "low_margin.jsonl"), low_margin_records)
    write_jsonl(os.path.join(args.out_dir, "low_info_trace.jsonl"), low_info_records)

    # report.md
    report = []
    report.append(f"# DRAMA-X CoC QA Gate Report\n")
    report.append(f"- input: `{args.input}`\n")
    report.append(f"- total: **{total}**\n")
    report.append(f"- low_margin_thr: **{args.low_margin_thr}**\n")
    report.append(f"- low_margin_count: **{len(low_margin_records)}**\n")
    report.append(f"- low_info_count: **{len(low_info_records)}**\n\n")

    report.append("## Risk label distribution\n")
    for k, v in risk_counter.most_common():
        report.append(f"- {k}: {v}\n")

    report.append("\n## Driving decision distribution\n")
    for k, v in decision_counter.most_common():
        report.append(f"- {k}: {v}\n")

    report.append("\n## Consistency (heuristic)\n")
    for k, v in consistency_counter.most_common():
        report.append(f"- {k}: {v}\n")

    report.append("\n## Trace length (chars)\n")
    report.append(f"- min: {stats['trace_len_min']}\n")
    report.append(f"- p50: {stats['trace_len_p50']}\n")
    report.append(f"- p90: {stats['trace_len_p90']}\n")
    report.append(f"- max: {stats['trace_len_max']}\n")

    report.append("\n## Next actions (recommended)\n")
    report.append("- Manually review `samples_for_review.jsonl` (random + low-margin + low-info).\n")
    report.append("- If low-margin samples often have wrong primary_vru: consider v3 (temporal / relative-motion cues).\n")
    report.append("- If low-info traces dominate: selective re-annotation with larger model on low-info subset.\n")

    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"[OK] wrote: {args.out_dir}/stats.json")
    print(f"[OK] wrote: {args.out_dir}/report.md")
    print(f"[OK] wrote: {args.out_dir}/samples_for_review.jsonl")
    print(f"[OK] wrote: {args.out_dir}/low_margin.jsonl")
    print(f"[OK] wrote: {args.out_dir}/low_info_trace.jsonl")


if __name__ == "__main__":
    main()
