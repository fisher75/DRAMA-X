import json
from pathlib import Path
from typing import Dict, Tuple, List

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
)


def load_gt_risk(jsonl_path: str) -> Dict[str, str]:
    """
    读取 DRAMA-X_hf/drama_x_annotations_populated.jsonl
    返回: { sample_id (id) -> 'Yes' / 'No' }
    """
    gt: Dict[str, str] = {}
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            # 有的版本用 "sample_id"，有的用 "id"
            sid = sample.get("sample_id", sample.get("id"))
            if sid is None:
                raise ValueError(f"No id/sample_id in sample keys={sample.keys()}")
            label = sample.get("Risk")
            if label is None:
                raise ValueError(f"No 'Risk' field in sample id={sid}")
            gt[sid] = label
    return gt


def load_pred_risk(json_path: str) -> Dict[str, str]:
    """
    读取预测结果 qwen3_2b_pred_risk.json
    结构假定为: { sample_id -> 'Yes' / 'No' 或 0/1 }

    如果你之后改成 list[dict] 形式（每个元素包含 sample_id 和 risk），
    这里也做了兼容处理。
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        preds = json.load(f)

    # 兼容 list[dict] 的写法
    if isinstance(preds, list):
        tmp = {}
        for item in preds:
            sid = item.get("sample_id")
            if sid is None:
                continue
            tmp[sid] = item.get("risk_pred_label", item.get("Risk"))
        preds = tmp

    return preds


def label_to_int(label) -> int:
    """
    把各种形式的标签映射成 0/1:
    - 'Yes' / 'yes' / '1' / True  -> 1
    - 'No'  / 'no'  / '0' / False -> 0
    """
    if isinstance(label, bool):
        return int(label)

    if isinstance(label, (int, float)):
        # 假设已经是 0/1
        return int(round(label))

    if isinstance(label, str):
        s = label.strip().lower()
        if s in {"yes", "y", "1", "true", "risk", "risky", "hazard"}:
            return 1
        if s in {"no", "n", "0", "false", "safe", "none"}:
            return 0

    raise ValueError(f"Unrecognized risk label: {label!r}")


def evaluate_risk(
    gt_risk: Dict[str, str],
    pred_risk: Dict[str, str],
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    对齐 GT 和预测，计算 Risk 的各项指标。

    返回:
      metrics: 指标字典
      y_true, y_pred: 对应的 0/1 列表（方便你后续画图/分析）
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    missing_pred = 0
    extra_pred = 0

    for sid, gt_label in gt_risk.items():
        if sid not in pred_risk:
            missing_pred += 1
            continue

        try:
            y_true.append(label_to_int(gt_label))
            y_pred.append(label_to_int(pred_risk[sid]))
        except ValueError:
            # 如果有无法解析的标签，直接跳过该样本
            continue

    # 统计 prediction 中多出的样本数量（通常是 GT 没有的 id）
    for sid in pred_risk.keys():
        if sid not in gt_risk:
            extra_pred += 1

    if not y_true:
        raise RuntimeError("No overlapping samples between GT and predictions!")

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "num_gt_samples": len(gt_risk),
        "num_pred_samples": len(pred_risk),
        "num_overlap": len(y_true),
        "num_missing_pred": missing_pred,
        "num_extra_pred": extra_pred,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_pos": precision,
        "recall_pos": recall,
        "f1_pos": f1,
        "confusion_matrix_[[TN,FP],[FN,TP]]": cm,
    }
    return metrics, y_true, y_pred


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate risk prediction on DRAMA-X (Yes/No)."
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="../DRAMA-X_hf/drama_x_annotations_populated.jsonl",
        help="Path to drama_x_annotations_populated.jsonl",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="../drama_intent/outputs/qwen3_local/qwen3_2b_pred_risk.json",
        help="Path to qwen3_2b_pred_risk.json (sample_id -> Yes/No or 0/1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="risk_eval_qwen3_2b.json",
        help="Where to save evaluation results (JSON).",
    )

    args = parser.parse_args()

    gt_risk = load_gt_risk(args.gt)
    pred_risk = load_pred_risk(args.pred)

    metrics, y_true, y_pred = evaluate_risk(gt_risk, pred_risk)

    # 打印主要指标
    print("===== Risk Evaluation (binary Yes/No) =====")
    for k in [
        "num_gt_samples",
        "num_pred_samples",
        "num_overlap",
        "num_missing_pred",
        "num_extra_pred",
        "accuracy",
        "balanced_accuracy",
        "precision_pos",
        "recall_pos",
        "f1_pos",
        "confusion_matrix_[[TN,FP],[FN,TP]]",
    ]:
        print(f"{k}: {metrics[k]}")

    # 保存结果到文件
    out_path = Path(args.out)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {out_path.resolve()}")


if __name__ == "__main__":
    main()
