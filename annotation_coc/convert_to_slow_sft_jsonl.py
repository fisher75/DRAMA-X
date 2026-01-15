#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DRAMA-X CoC jsonl to slow-system SFT jsonl.
Outputs jsonl with {sample_id, image(optional), conversations:[{from,value},...]}.
Only stdlib.

python convert_to_slow_sft_jsonl.py \
  --input /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b_v2_full_rule_based_20260108.jsonl \
  --output /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_slow_sft_v2_rule.jsonl \
  --include_structured_facts \
  --topk_vru 3 \
  --prompt_lang en
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as e:
                raise RuntimeError(f"[JSON parse fail] {path}:{ln} -> {e}")


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_sample_id(r: Dict[str, Any]) -> str:
    return str(safe_get(r, ["sample_id", "id", "clip_id", "uid"], "UNKNOWN"))


def get_image_path(r: Dict[str, Any]) -> Optional[str]:
    # adapt if you store local paths; otherwise keep None
    return safe_get(r, ["image", "image_path", "frame_path", "img_path"], None)


def get_primary_vru(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cc = r.get("critical_components") or {}
    if isinstance(cc, dict) and isinstance(cc.get("primary_vru"), dict):
        return cc["primary_vru"]
    if isinstance(r.get("primary_vru"), dict):
        return r["primary_vru"]
    return None


def get_vru_list(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = r.get("vru_list") or r.get("vrus") or r.get("objects") or []
    return v if isinstance(v, list) else []


def get_decision(r: Dict[str, Any]) -> str:
    return str(safe_get(r, ["driving_decision", "decision", "action"], "UNKNOWN")).upper()


def get_trace(r: Dict[str, Any]) -> str:
    return str(safe_get(r, ["coc_trace", "trace", "reasoning", "explanation"], "")).strip()


def vru_brief(v: Dict[str, Any]) -> str:
    vid = safe_get(v, ["id", "vru_id", "track_id"], "")
    vtype = safe_get(v, ["type", "category", "cls"], "")
    intent = safe_get(v, ["intent", "motion", "behavior"], "")
    pos = safe_get(v, ["position", "loc", "region"], "")
    score = safe_get(v, ["risk_score", "score"], None)
    parts = []
    if vid: parts.append(f"id={vid}")
    if vtype: parts.append(f"type={vtype}")
    if intent: parts.append(f"intent={intent}")
    if pos: parts.append(f"pos={pos}")
    if score is not None:
        try:
            parts.append(f"risk_score={float(score):.3f}")
        except:
            parts.append(f"risk_score={score}")
    return "{" + ", ".join(parts) + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--topk_vru", type=int, default=3)
    ap.add_argument("--include_structured_facts", action="store_true")
    ap.add_argument("--prompt_lang", choices=["en", "zh"], default="en")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    def build_prompt(primary: Optional[Dict[str, Any]], vrus: List[Dict[str, Any]]) -> str:
        if args.prompt_lang == "zh":
            base = (
                "你是自动驾驶风险推理助手。请基于图像进行外部风险分析，"
                "输出严格 JSON，字段包括：driving_decision (GO|SLOW|STOP), "
                "critical_components.primary_vru, coc_trace。\n"
            )
        else:
            base = (
                "You are an autonomous driving risk reasoning assistant. "
                "Based on the image, output a STRICT JSON with fields: "
                "driving_decision (GO|SLOW|STOP), critical_components.primary_vru, coc_trace.\n"
            )

        if not args.include_structured_facts:
            return base + "Return JSON only."

        facts = []
        if primary:
            facts.append("Primary VRU (rule-based): " + vru_brief(primary))
        if vrus:
            facts.append(f"Top-{min(args.topk_vru, len(vrus))} VRUs (sorted by risk):")
            for i, v in enumerate(vrus[: args.topk_vru], 1):
                facts.append(f"  {i}. {vru_brief(v)}")
        return base + "\n".join(facts) + "\nReturn JSON only."

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for ln, r in read_jsonl(args.input):
            sid = get_sample_id(r)
            img = get_image_path(r)  # can be None
            primary = get_primary_vru(r)
            vrus = get_vru_list(r)

            prompt = build_prompt(primary, vrus)

            # target is your existing CoC JSON
            target = {
                "driving_decision": get_decision(r),
                "critical_components": {"primary_vru": primary or {}},
                "coc_trace": get_trace(r),
            }

            item = {
                "sample_id": sid,
                "image": img,
                "conversations": [
                    {"from": "user", "value": prompt},
                    {"from": "assistant", "value": json.dumps(target, ensure_ascii=False)},
                ],
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
