import json
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
from bert_score import score as bert_score

import csv

def export_errors_for_analysis(data, out_csv="drama_x_errors.csv"):
    """
    导出典型错误样本，方便你肉眼看图和文本。
    假设 item 中有 image_path, gt, pred 字段。
    """
    rows = []
    for item in data:
        sid = item.get("sample_id", "")
        img = item.get("image_path", "")
        gt = item.get("gt", {})
        pred = item.get("pred", {})

        gt_int = gt.get("intent")
        pd_int = pred.get("intent")
        gt_risk = gt.get("risk")
        pd_risk = pred.get("risk")
        gt_act = gt.get("action")
        pd_act = pred.get("action")

        # 简单规则：只保留至少有一个任务预测错误的样本
        wrong = False
        if gt_int is not None and pd_int is not None and gt_int != pd_int:
            wrong = True
        if gt_risk is not None and pd_risk is not None and gt_risk != pd_risk:
            wrong = True
        if gt_act is not None and pd_act is not None and gt_act != pd_act:
            wrong = True

        if wrong:
            rows.append({
                "sample_id": sid,
                "image_path": img,
                "gt_intent": gt_int,
                "pred_intent": pd_int,
                "gt_risk": gt_risk,
                "pred_risk": pd_risk,
                "gt_action": gt_act,
                "pred_action": pd_act,
                "pred_reasoning": pred.get("reasoning", "")
            })

    # 只导出前 N 条，避免太大
    N = min(len(rows), 200)
    rows = rows[:N]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"导出了 {len(rows)} 条错误样本到 {out_csv}")


def load_results(json_path):
    """加载你 2B 推理生成的 JSON 文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data 可能是 list，也可能是 dict，自己按格式调整
    return data


def extract_labels(data):
    """
    从 JSON 中提取三类任务的 GT 和 Pred：
    IP（intent），RA（risk），AS（action）
    """
    gt_intents, pred_intents = [], []
    gt_risks, pred_risks = [], []
    gt_actions, pred_actions = [], []
    gt_action_texts, pred_action_texts = [], []  # 用于 BERTScore

    for item in data:
        gt = item.get("gt", {})
        pred = item.get("pred", {})

        # 1) Intent Prediction (IP)
        if "intent" in gt and "intent" in pred:
            gt_intents.append(gt["intent"])
            pred_intents.append(pred["intent"])

        # 2) Risk Assessment (RA)
        if "risk" in gt and "risk" in pred:
            gt_risks.append(gt["risk"])
            pred_risks.append(pred["risk"])

        # 3) Action Suggestion (AS)
        # 这里考虑两种：分类标签 + 自然语言句子
        if "action" in gt and "action" in pred:
            gt_actions.append(gt["action"])
            pred_actions.append(pred["action"])

        # 如果你想用自然语言 BERTScore，就另外存 action_text
        if "action_text" in gt and "action_text" in pred:
            gt_action_texts.append(gt["action_text"])
            pred_action_texts.append(pred["action_text"])

    return {
        "ip": (gt_intents, pred_intents),
        "ra": (gt_risks, pred_risks),
        "as_cls": (gt_actions, pred_actions),
        "as_text": (gt_action_texts, pred_action_texts),
    }


def eval_ip(gt_intents, pred_intents):
    print("=== Intent Prediction (IP) ===")
    print("样本数:", len(gt_intents))
    print("Accuracy:", accuracy_score(gt_intents, pred_intents))
    print("Macro F1:", f1_score(gt_intents, pred_intents, average="macro"))
    print("分类报告:\n", classification_report(gt_intents, pred_intents))


def eval_ra(gt_risks, pred_risks):
    print("=== Risk Assessment (RA) ===")
    print("样本数:", len(gt_risks))
    print("Accuracy:", accuracy_score(gt_risks, pred_risks))
    print("Balanced Accuracy:", balanced_accuracy_score(gt_risks, pred_risks))
    print("Macro F1:", f1_score(gt_risks, pred_risks, average="macro"))
    print("分类报告:\n", classification_report(gt_risks, pred_risks))


def eval_as_cls(gt_actions, pred_actions):
    print("=== Action Suggestion (AS) - Label Accuracy ===")
    print("样本数:", len(gt_actions))
    print("Accuracy:", accuracy_score(gt_actions, pred_actions))
    print("Macro F1:", f1_score(gt_actions, pred_actions, average="macro"))
    print("分类报告:\n", classification_report(gt_actions, pred_actions))


def eval_as_bert(gt_texts, pred_texts):
    if len(gt_texts) == 0:
        print("没有 action_text 字段，跳过 BERTScore")
        return
    print("=== Action Suggestion (AS) - BERTScore ===")
    P, R, F1 = bert_score(pred_texts, gt_texts, lang="en", rescale_with_baseline=True)
    print("BERTScore-F1 (mean):", float(F1.mean()))
    # 你也可以输出 P/R，或者后面再细化


def main():
    json_path = "your_qwen2b_drama_x_results.json"  # TODO: 换成你的文件路径
    data = load_results(json_path)

    labels = extract_labels(data)
    gt_ip, pred_ip = labels["ip"]
    gt_ra, pred_ra = labels["ra"]
    gt_as, pred_as = labels["as_cls"]
    gt_as_text, pred_as_text = labels["as_text"]

    if len(gt_ip) > 0:
        eval_ip(gt_ip, pred_ip)
    if len(gt_ra) > 0:
        eval_ra(gt_ra, pred_ra)
    if len(gt_as) > 0:
        eval_as_cls(gt_as, pred_as)
    if len(gt_as_text) > 0:
        eval_as_bert(gt_as_text, pred_as_text)
    
    export_errors_for_analysis(data, out_csv="/workspace/chz/code/DRAMA-X/drama_x_errors.csv")


if __name__ == "__main__":
    main()
    
