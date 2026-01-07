#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_coc_failures.py

用途（一个脚本包含你需要的所有功能）：
1) 覆盖率核对：structured / ok(v2) / fail / missing
2) 合并 fail：rank*.fail.jsonl -> fail_merged.jsonl（可选）
3) 统计 fail 主因：reason TopN、stage TopN、reason×stage TopN
4) 风险分布分析：risk_label / vru_count 与失败关系（TopN）
5) 导出 missing_line_index.jsonl：便于 retry / 定位

建议运行：
python analyze_coc_failures.py \
  --structured /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_structured.jsonl \
  --ok /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b_v2.jsonl \
  --fail_glob "/workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b_v2.jsonl.rank*.fail.jsonl" \
  --out_dir /workspace/chz/code/DRAMA-X/annotation_coc/_analysis_v2 \
  --topk 30
"""

import os
import sys
import json
import glob
import argparse
from collections import Counter, defaultdict


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # 保留行号信息，方便定位
                yield {"_bad_json_line": i, "_raw": line[:200]}


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured", required=True, help="drama_x_structured.jsonl")
    ap.add_argument("--ok", required=True, help="v2 ok output jsonl")
    ap.add_argument("--fail_glob", required=True, help='glob like "...rank*.fail.jsonl"')
    ap.add_argument("--out_dir", required=True, help="output directory for reports")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--write_fail_merged", action="store_true",
                    help="Write merged fail jsonl to out_dir/fail_merged.jsonl")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    structured_path = args.structured
    ok_path = args.ok
    fail_paths = sorted(glob.glob(args.fail_glob))

    if not fail_paths:
        print(f"[ERROR] No fail files matched fail_glob: {args.fail_glob}")
        sys.exit(1)

    print("==== Inputs ====")
    print("structured:", structured_path)
    print("ok:", ok_path)
    print("fail_files:", len(fail_paths))
    for fp in fail_paths[:10]:
        print("  -", fp)
    if len(fail_paths) > 10:
        print("  ...")

    # ---- 1) 统计 structured 行数（以及 line_index 的全集）----
    structured_indices = []
    structured_count = 0
    for idx, _ in enumerate(open(structured_path, "r", encoding="utf-8")):
        structured_count += 1
        structured_indices.append(idx)
    structured_set = set(structured_indices)

    # ---- 2) 读取 ok 输出（如果 preserve_order 用过，这里不依赖 _line_index）----
    ok_count = 0
    ok_sample_ids = set()
    ok_line_index_set = set()

    for obj in read_jsonl(ok_path):
        ok_count += 1
        sid = obj.get("sample_id", obj.get("id", None))
        if sid is not None:
            ok_sample_ids.add(str(sid))
        # 有些版本可能保留了 line_index 或 _line_index（你 v2 merge 后会删 _line_index）
        li = obj.get("line_index", None)
        if li is None:
            li = obj.get("_line_index", None)
        li = safe_int(li, default=None)
        if li is not None:
            ok_line_index_set.add(li)

    # ---- 3) 合并 fail：统计 reason / stage / risk_label / vru_count / line_index ----
    fail_count = 0
    fail_reason = Counter()
    fail_stage = Counter()
    fail_reason_stage = Counter()

    fail_risk = Counter()
    fail_vru_count = Counter()
    fail_line_index_set = set()

    merged_fail_path = os.path.join(args.out_dir, "fail_merged.jsonl")
    fout_fail = open(merged_fail_path, "w", encoding="utf-8") if args.write_fail_merged else None

    for fp in fail_paths:
        for obj in read_jsonl(fp):
            # 跳过坏 json 行
            if "_bad_json_line" in obj:
                fail_count += 1
                fail_reason["bad_json_line"] += 1
                continue

            fail_count += 1
            reason = obj.get("reason", "unknown")
            stage = obj.get("stage", "unknown")
            fail_reason[reason] += 1
            fail_stage[stage] += 1
            fail_reason_stage[(reason, stage)] += 1

            risk = obj.get("risk_label", "unknown")
            fail_risk[str(risk)] += 1

            vc = obj.get("vru_count", "unknown")
            fail_vru_count[str(vc)] += 1

            li = safe_int(obj.get("line_index", None), default=None)
            if li is not None:
                fail_line_index_set.add(li)

            if fout_fail is not None:
                fout_fail.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if fout_fail is not None:
        fout_fail.close()
        print(f"[OK] wrote merged fail -> {merged_fail_path}")

    # ---- 4) missing 估算 ----
    # 最可靠：structured - ok_by_line_index - fail_by_line_index
    # 但 ok 输出一般没带 line_index（merge 后删除了 _line_index），所以：
    #   - missing_by_index 主要用于 “fail 是否覆盖全部失败”
    #   - 真正缺失以 (structured_count - ok_count - fail_count) 为主
    missing_est = structured_count - ok_count - fail_count

    print("\n==== Coverage Summary ====")
    print(f"structured_total = {structured_count}")
    print(f"ok_total         = {ok_count}")
    print(f"fail_total       = {fail_count}")
    print(f"missing_est      = {missing_est}  (structured - ok - fail)")

    # 如果 fail_line_index_set 能覆盖 structured_set 的很大部分，说明 fail log 写得完整
    # 这里导出 fail line_index 和 missing line_index（按索引角度）
    missing_index_path = os.path.join(args.out_dir, "missing_line_index.jsonl")

    # 这里的 missing_by_index 是：structured index 里既不在 fail 也不在 ok_line_index_set
    # 由于 ok_line_index_set 通常为空，这个只对你“保留 line_index 的 ok 文件”有意义
    missing_by_index = sorted(list(structured_set - fail_line_index_set - ok_line_index_set))

    with open(missing_index_path, "w", encoding="utf-8") as f:
        for li in missing_by_index:
            f.write(json.dumps({"line_index": li}, ensure_ascii=False) + "\n")

    print(f"[OK] wrote missing_line_index.jsonl -> {missing_index_path}")
    print(f"missing_by_index_count = {len(missing_by_index)} (note: ok file may not carry line_index)")

    # ---- 5) 写统计报告 ----
    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        def w(s=""):
            f.write(s + "\n")

        w("==== Coverage Summary ====")
        w(f"structured_total = {structured_count}")
        w(f"ok_total         = {ok_count}")
        w(f"fail_total       = {fail_count}")
        w(f"missing_est      = {missing_est}  (structured - ok - fail)")
        w("")

        w(f"==== Top {args.topk} Fail Reasons ====")
        for k, v in fail_reason.most_common(args.topk):
            w(f"{v}\t{k}")
        w("")

        w(f"==== Top {args.topk} Fail Stages ====")
        for k, v in fail_stage.most_common(args.topk):
            w(f"{v}\t{k}")
        w("")

        w(f"==== Top {args.topk} (Reason × Stage) ====")
        for (reason, stage), v in fail_reason_stage.most_common(args.topk):
            w(f"{v}\t{reason}\t{stage}")
        w("")

        w(f"==== Fail Risk Label Distribution ====")
        for k, v in fail_risk.most_common():
            w(f"{v}\t{k}")
        w("")

        w(f"==== Fail VRU Count Distribution ====")
        for k, v in fail_vru_count.most_common(30):
            w(f"{v}\t{k}")
        w("")

    print(f"[OK] wrote report -> {report_path}")

    # 同时在屏幕上也打印关键 TopK
    print("\n==== Top Fail Reasons ====")
    for k, v in fail_reason.most_common(min(args.topk, 15)):
        print(v, k)

    print("\n==== Top Fail Stages ====")
    for k, v in fail_stage.most_common(min(args.topk, 15)):
        print(v, k)

    print("\n==== Top (Reason × Stage) ====")
    for (reason, stage), v in fail_reason_stage.most_common(min(args.topk, 15)):
        print(v, reason, " / ", stage)

    print("\n==== Done ====")
    print("Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
