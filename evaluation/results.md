DRAMA-X Baseline v0（Qwen3-VL-2B zero-shot, end-to-end）:
(/workspace/conda_envs/dramax) haozhuang@colab-automan:/workspace/chz/code/DRAMA-X/evaluation$ python ./run_eval_intent.py 
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4381/4381 [00:00<00:00, 241731.84it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4381/4381 [00:00<00:00, 282600.44it/s]
===== [A] End-to-end BBox Evaluation (预测框) =====
{'overall_accuracy': 0.00591304347826087, 'class_accuracies': {'pedestrian': 0.0061773255813953485, 'cyclist': 0.0}, 'total_objects': 5750, 'correct_detections': 34}

===== [A] End-to-end Intent Evaluation (预测框 + 预测意图) =====
{'overall_accuracy': 0.0, 'horizontal_accuracy': 0.0, 'vertical_accuracy': 0.0, 'total_intents': 11720, 'correct_intents': 0, 'total_horizontal': 5860, 'correct_horizontal': 0, 'total_vertical': 5860, 'correct_vertical': 0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4381/4381 [00:00<00:00, 446196.05it/s]

===== [B] Intent Evaluation with GT BBoxes (skip_bbox_matching=True) =====
{'overall_accuracy': 0.24197952218430033, 'horizontal_accuracy': 0.3916382252559727, 'vertical_accuracy': 0.09232081911262799, 'total_intents': 11720, 'correct_intents': 2836, 'total_horizontal': 5860, 'correct_horizontal': 2295, 'total_vertical': 5860, 'correct_vertical': 541}

分析：
1. A 组：端到端（预测 bbox + 预测 intent）
[BBox] overall_accuracy = 0.0059  (~0.6%)
      pedestrian = 0.0062, cyclist = 0.0
      total_objects = 5750, correct_detections = 34

[Intent] overall_accuracy = 0
         horizontal_accuracy = 0
         vertical_accuracy   = 0

1）BBox
数据里一共 5750 个 VRU GT box，Qwen 只对了 34 个（IoU ≥ 0.25）。
pedestrian 0.62%，cyclist 直接 0。
👉 说明：让 Qwen3-VL-2B 直接从原图画 bbox，几乎完全不可用。
这和 DRAMA-X 原文结论高度一致：VLM 在“精确定位”这件事上非常差。

2）Intent = 0 的原因
od_intent_eval 的逻辑是：
先用 IoU ≥ 0.6 找到「预测框 ↔ GT 框」的匹配；
只对这些匹配上的 pair 去比较 Intent[0] / Intent[1] 文本是否完全相同。
你的 bbox 本来就几乎全挂（IoU≥0.25 只有 0.6%，更别说 0.6），
所以几乎 没有任何 GT VRU 能找到 IoU≥0.6 的预测框，
自然 Intent 统计出来就是：
total_intents 很多，但匹配到的 pair≈0，correct_intents=0。
👉 这不是“模型一点意图都不会”；而是：
“在找不到对的框” 的前提下，Intent 根本没有发挥空间。

2. B 组：用 GT bbox，只评估 Intent
[B] Intent (GT bboxes, skip_bbox_matching=True)
overall_accuracy     = 0.242
horizontal_accuracy  = 0.392
vertical_accuracy    = 0.092
total_intents        = 11720
correct_intents      = 2836


这里我们告诉 od_intent_gt_eval.evaluate_intents：
“不要再看预测 bbox 了，直接用 GT 的 box 来对齐目标，只看 Qwen 预测的 Intent 文本对不对。”
因此：
horizontal_accuracy：水平方向（left / right / stationary）是否预测正确。
→ 约 39%，说明模型对 “往左/往右/不动” 这一维有一定辨别能力，比纯随机要好不少。
vertical_accuracy：纵向（towards / away / stationary）是否正确。
→ 只有 9% 左右，几乎接近“乱猜”的水平。
overall_accuracy：同时把横向 + 纵向都预测对的比例。
→ 约 24%，可以理解成「模型有时能抓住一个方向，但两维都一起对就很难」。
直觉上也合理：
横向移动（向左/向右）在图像上是 像素平面内的位移，视觉 cues 比较明显；
“towards/away ego vehicle” 涉及 深度变化、尺寸变化、透视关系，
对 2B 级别、未专门训过此任务的 VLM 来说难度很高，所以基本瞎。

3. 对你的项目意味着什么？
这组结果其实非常有价值，几乎可以直接写进未来 paper / PPT 的「动机」里：
Fast system 必须是专门的感知模型
端到端让 VLM 负责 bbox → 几乎零分；
这为你“外部用 ODS/Tracking，内部用 VLM reasoning”提供了硬数据支持。
Slow system（意图/风险推理）有一定基础，但不够好
在「已知 bbox」条件下：
水平方向 39%：说明 Qwen3-VL-2B 作为“意图分类器”是有点用的，
但远称不上好模型，有很大 Fine-tune / CoC 提升空间。
纵向只有 9%：
在 risk reasoning 里，“towards ego” 是非常关键的 risk indicator；
现在几乎不会 → 这正好说明你需要：
引入 物理/几何特征（相对速度、TTC）；
或者用类似 Alpamayo-R1 那种更结构化的 CoC 标注去引导。
“Intent only” vs “End-to-end” 的对比图，可以直接变成 Methodology 里的一个小 figure
A：bbox+intent 一起评 → 完全挂；
B：GT bbox → Intent 还能勉强 24% / 39% / 9%；
这恰好展示了你想做的两级系统结构：
Stage 1：外部专用感知 + 内部专用 DMS
Stage 2：把高质量 ROI + 场景信息交给 VLM 做 Unified Reasoning