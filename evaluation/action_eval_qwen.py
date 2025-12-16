import json
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def load_gt_actions(jsonl_path: str) -> Dict[str, str]:
    """
    从 DRAMA-X_hf/drama_x_annotations_populated.jsonl 里读取 GT 动作：
      { sample_id -> suggested_action (str) }
    """
    gt: Dict[str, str] = {}
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            sid = sample.get("sample_id", sample.get("id"))
            if sid is None:
                raise ValueError(f"No id/sample_id in sample keys={sample.keys()}")
            act = sample.get("suggested_action")
            if act is None:
                # 理论上不该发生，防御性代码
                continue
            gt[sid] = act
    return gt


def load_pred_actions(json_path: str) -> Dict[str, str]:
    """
    读取预测结果 qwen3_2b_pred_actions.json

    支持两种格式：
    1) { sample_id: "action text" }
    2) [ { "sample_id": "...", "Suggested_action": "..." }, ... ]
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # dict 形式
    if isinstance(data, dict):
        return {sid: str(act) for sid, act in data.items()}

    # list[dict] 形式
    if isinstance(data, list):
        out: Dict[str, str] = {}
        for item in data:
            sid = item.get("sample_id") or item.get("id")
            if sid is None:
                continue
            act = item.get("Suggested_action") or item.get("suggested_action")
            if act is None:
                continue
            out[sid] = str(act)
        return out

    raise ValueError(f"Unsupported prediction file format: {type(data)}")


def normalize_action_class(text: str) -> str:
    """
    把自由文本的动作归一到 3 类（+1 个 NA）：
      - 'STOP'   : 包含 stop / halt / yield
      - 'SLOW'   : 包含 slow / cautious / careful / manoeuvre / aware
      - 'GO'     : 包含 accelerate / start moving / follow / continue / maintain / proceed / drive
      - 'NA'     : 明确 'N/A'，评测时会被丢弃

    其它没匹配上的，暂时归到 'SLOW' or 'GO' 都可以，这里倾向归到 'SLOW' 以偏保守。
    """
    if text is None:
        return "NA"

    s = text.strip().lower()

    # 明确的 NA
    if s in {"n/a", "na"}:
        return "NA"

    # STOP 类
    if "stop" in s or "halt" in s or "yield" in s:
        return "STOP"

    # SLOW / 谨慎 / 绕行 / 提醒注意
    if (
        "slow" in s
        or "cautious" in s
        or "careful" in s
        or "manoeuvre" in s
        or "maneuver" in s
        or "aware" in s
        or "be aware" in s
    ):
        return "SLOW"

    # GO / 继续 / 加速 / 跟车
    if (
        "accelerate" in s
        or "start moving" in s
        or "follow" in s
        or "continue" in s
        or "maintain" in s
        or "proceed" in s
        or "drive" in s
        or "go " in s
        or s == "go"
    ):
        return "GO"

    # 兜底：不认识的动作用 SLOW 处理（保守一点）
    return "SLOW"


def build_label_pairs(
    gt_actions: Dict[str, str],
    pred_actions: Dict[str, str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    对齐 GT 和预测，返回：
      y_true_cls, y_pred_cls, y_true_text, y_pred_text

    - 只保留 GT / Pred 都存在的 sample_id
    - 丢弃 GT 动作为 NA 的样本（即‘N/A’）
    """
    y_true_cls: List[str] = []
    y_pred_cls: List[str] = []
    y_true_text: List[str] = []
    y_pred_text: List[str] = []

    num_missing_pred = 0
    for sid, gt_text in gt_actions.items():
        if sid not in pred_actions:
            num_missing_pred += 1
            continue
        pred_text = pred_actions[sid]

        gt_cls = normalize_action_class(gt_text)
        pred_cls = normalize_action_class(pred_text)

        # 丢弃 GT 为 'NA' 的样本
        if gt_cls == "NA":
            continue

        y_true_cls.append(gt_cls)
        y_pred_cls.append(pred_cls)
        y_true_text.append(gt_text)
        y_pred_text.append(pred_text)

    print(f"[Info] Missing predictions for {num_missing_pred} GT samples.")
    print(f"[Info] Used {len(y_true_cls)} samples for action classification eval.")
    return y_true_cls, y_pred_cls, y_true_text, y_pred_text


def evaluate_actions(
    y_true_cls: List[str],
    y_pred_cls: List[str],
) -> Dict:
    """
    计算 STOP/SLOW/GO 的分类指标。
    """
    if not y_true_cls:
        raise RuntimeError("No samples to evaluate (y_true_cls is empty).")

    labels_sorted = sorted(set(y_true_cls))  # e.g. ['GO','SLOW','STOP']
    acc = accuracy_score(y_true_cls, y_pred_cls)
    macro_f1 = f1_score(y_true_cls, y_pred_cls, average="macro")
    weighted_f1 = f1_score(y_true_cls, y_pred_cls, average="weighted")

    report = classification_report(
        y_true_cls,
        y_pred_cls,
        labels=labels_sorted,
        target_names=labels_sorted,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels_sorted).tolist()

    metrics = {
        "classes": labels_sorted,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    return metrics


def evaluate_bert_score(
    y_true_text: List[str],
    y_pred_text: List[str],
) -> Dict:
    """
    可选：计算动作文本的 BERTScore。
    如果环境中没有 bert-score，则返回 {'available': False}。
    """
    try:
        from bert_score import score as bertscore
    except ImportError:
        print("[Warn] bert-score not installed, skip BERTScore evaluation.")
        return {"available": False}

    if not y_true_text:
        return {"available": False}

    # bert_score: cands, refs
    P, R, F1 = bertscore(
        y_pred_text,
        y_true_text,
        lang="en",
        rescale_with_baseline=True,
    )
    # 转成 Python float
    bert_metrics = {
        "available": True,
        "precision_mean": float(P.mean().item()),
        "recall_mean": float(R.mean().item()),
        "f1_mean": float(F1.mean().item()),
    }
    return bert_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Action Suggestion (Suggested_action) on DRAMA-X."
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
        default="../drama_intent/outputs/qwen3_local/qwen3_2b_pred_actions.json",
        help="Path to qwen3_2b_pred_actions.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="action_eval_qwen3_2b.json",
        help="Where to save evaluation results (JSON).",
    )

    args = parser.parse_args()

    gt_actions = load_gt_actions(args.gt)
    pred_actions = load_pred_actions(args.pred)

    y_true_cls, y_pred_cls, y_true_text, y_pred_text = build_label_pairs(
        gt_actions, pred_actions
    )

    metrics_cls = evaluate_actions(y_true_cls, y_pred_cls)
    metrics_bert = evaluate_bert_score(y_true_text, y_pred_text)

    # 打印主要指标
    print("===== Action Suggestion Evaluation (class-level STOP/SLOW/GO) =====")
    print(f"classes: {metrics_cls['classes']}")
    print(f"accuracy: {metrics_cls['accuracy']:.4f}")
    print(f"macro_f1: {metrics_cls['macro_f1']:.4f}")
    print(f"weighted_f1: {metrics_cls['weighted_f1']:.4f}")
    print("confusion_matrix (rows=GT, cols=Pred):")
    print(metrics_cls["confusion_matrix"])

    if metrics_bert.get("available", False):
        print("\n===== BERTScore (text-level similarity) =====")
        print(f"precision_mean: {metrics_bert['precision_mean']:.4f}")
        print(f"recall_mean   : {metrics_bert['recall_mean']:.4f}")
        print(f"f1_mean       : {metrics_bert['f1_mean']:.4f}")
    else:
        print("\n[Info] BERTScore not evaluated (bert-score not installed).")

    # 保存到 JSON
    out = {
        "class_metrics": metrics_cls,
        "bert_score": metrics_bert,
        "num_samples_used": len(y_true_cls),
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {out_path.resolve()}")


if __name__ == "__main__":
    main()
