````markdown
# DRAMA-X + Qwen3-VL-2B 基线实验说明（TODO1）

## 0. 总体目标

本目录记录在 **DRAMA-X** 数据集上，使用 **Qwen3-VL-2B-Instruct** 做的第一版基线实验（TODO1），包括三条任务：

- **Intent Prediction (IP)**：行人/骑行者的意图预测（左右 + 朝向）
- **Risk Assessment (RA)**：当前帧是否存在 VRU 风险（Risk Yes/No）
- **Action Suggestion (AS)**：系统建议的驾驶动作（STOP / SLOW / GO）

并给出：

1. 推理与评测相关脚本的使用方法；
2. 当前 zero-shot 基线的指标总结；
3. 对 failure mode 和后续改进方向的初步分析。

---

## 1. 目录结构（与本 README 相关的部分）

仓库根目录假设为：

```bash
/workspace/chz/code/DRAMA-X/
````

关键子目录与文件：

```text
DRAMA-X/
├── DRAMA-X_hf/
│   ├── drama_x_annotations_populated.jsonl      # 官方 GT 标注（用于评测）
│   └── drama_x_annotated.jsonl                  # 原始标注版本（通常不用直接改）
│
├── drama_intent/
│   └── outputs/
│       └── qwen3_local/
│           ├── all_raw_Qwen3-VL-2B-Instruct_onepass.json
│           │      # Qwen3-VL-2B-Instruct 的原始推理输出（端到端）
│           ├── qwen3_2b_pred_intent_bbox.json  # 从 raw 提取后的 Intent+BBox 预测
│           ├── qwen3_2b_pred_risk.json         # 从 raw 提取后的 Risk 预测
│           └── qwen3_2b_pred_actions.json      # 从 raw 提取后的 Action 文本预测
│
├── evaluation/
│   ├── run_eval_intent.py        # Intent + BBox 评测（端到端 / GT bbox）
│   ├── risk_eval_qwen.py         # Risk Yes/No 评测
│   └── action_eval_qwen.py       # Action Suggestion 评测（STOP/SLOW/GO + BERTScore）
│
└── qwen3_local_sgg_intent_singleGPU.py  # 本地跑 Qwen3-VL-2B 的主推理脚本
└── run_multi_gpu.py                     # 多卡版本（可选）
└── convert_qwen3_raw_to_unified.py      # 从 raw JSON 生成统一预测文件（自写工具脚本）
```

> 注：`convert_qwen3_raw_to_unified.py` 是目前为了评测方便新增的工具脚本，用来把原始 Qwen 输出规整成评测脚本所需的 JSON 格式。

---

## 2. 环境依赖（简要）

* Python 3.10+（当前 `conda_envs/dramax`）
* 主要依赖：

  * `transformers`（Qwen3-VL 推理）
  * `torch` + GPU 驱动
  * `scikit-learn`（评测指标）
  * `bert-score`（可选，用于 Action 文本相似度）

示例安装（仅供参考，具体以当前环境为准）：

```bash
pip install transformers accelerate
pip install scikit-learn
pip install bert-score   # 可选
```

---

## 3. 推理阶段：生成原始 Qwen3-VL-2B 输出

### 3.1 单卡推理脚本：`qwen3_local_sgg_intent_singleGPU.py`

作用：
对 DRAMA-X 的每个样本，调用 **Qwen3-VL-2B-Instruct**，生成：

* 帧级：

  * `Risk`：`"Yes"` / `"No"`
  * `Suggested_action`：自然语言动作建议
* 目标级（如 `pedestrian_1`, `cyclist_1` 等）：

  * `Intent`：包含水平和纵向信息的 list（e.g. `["goes to the left", "moves away from ego vehicle"]`）
  * `Reason`：简短文字解释
  * `Bounding_box`：目标框 `[x1, y1, x2, y2]`

输出文件示例路径：

```text
drama_intent/outputs/qwen3_local/all_raw_Qwen3-VL-2B-Instruct_onepass.json
```

运行示例（伪命令，仅示意）：

```bash
cd /workspace/chz/code/DRAMA-X

python qwen3_local_sgg_intent_singleGPU.py \
  --model_name Qwen/Qwen3-VL-2B-Instruct \
  --output_path drama_intent/outputs/qwen3_local/all_raw_Qwen3-VL-2B-Instruct_onepass.json \
  --batch_size 1 \
  --device cuda:0
```

> 实际参数以脚本内默认值为准，这里主要说明数据流向与输出结构。

### 3.2 多卡脚本：`run_multi_gpu.py`（可选）

如果要在多张 GPU 上并行跑 Qwen 推理，可以用这个脚本进行拆分和并行。
目前 baseline 阶段主要是单卡版本跑通，因此此处略，后续按需要扩展。

---

## 4. 结果转换：从 raw JSON → 评测用预测文件

### 4.1 转换脚本：`convert_qwen3_raw_to_unified.py`

作用：
把 `all_raw_Qwen3-VL-2B-Instruct_onepass.json` 解析成三个评测用 JSON：

1. **Intent + BBox 预测**（给 `run_eval_intent.py` 用）
   `qwen3_2b_pred_intent_bbox.json`

   ```json
   {
     "clip_305_000786_frame_000786": {
       "pedestrian_1": {
         "Intent": ["goes to the right", "stationary"],
         "Reason": "...",
         "Bounding_box": [375, 513, 416, 642]
       },
       "car_1": {...}
     },
     ...
   }
   ```

2. **Risk 预测**（给 `risk_eval_qwen.py` 用）
   `qwen3_2b_pred_risk.json`

   ```json
   {
     "clip_305_000786_frame_000786": "No",
     "clip_1111_000552_frame_000552": "Yes",
     ...
   }
   ```

3. **Action 文本预测**（给 `action_eval_qwen.py` 用）
   `qwen3_2b_pred_actions.json`

   ```json
   {
     "clip_305_000786_frame_000786": "maintain current speed and position",
     ...
   }
   ```

运行示例：

```bash
cd /workspace/chz/code/DRAMA-X

python convert_qwen3_raw_to_unified.py \
  --input drama_intent/outputs/qwen3_local/all_raw_Qwen3-VL-2B-Instruct_onepass.json \
  --output_intent_bbox drama_intent/outputs/qwen3_local/qwen3_2b_pred_intent_bbox.json \
  --output_risk        drama_intent/outputs/qwen3_local/qwen3_2b_pred_risk.json \
  --output_actions     drama_intent/outputs/qwen3_local/qwen3_2b_pred_actions.json
```

> （实际参数名请与脚本内保持一致，这里强调输入输出关系。）

---

## 5. 评测脚本使用说明

评测统一在：

```bash
cd /workspace/chz/code/DRAMA-X/evaluation
```

### 5.1 Intent + BBox 评测：`run_eval_intent.py`

脚本位置：

```text
evaluation/run_eval_intent.py
```

功能：

* 加载 GT：`../DRAMA-X_hf/drama_x_annotations_populated.jsonl`
* 加载预测：`../drama_intent/outputs/qwen3_local/qwen3_2b_pred_intent_bbox.json`
* 评测两种设置：

  1. **End-to-end**：预测 BBox + 预测 Intent
  2. **GT BBox**：使用 GT BBox，只评估 Intent

运行：

```bash
python run_eval_intent.py
```

输出包含：

* `BBox` 评测：overall accuracy、per-class accuracy（pedestrian / cyclist）
* `Intent` 评测（e2e）：overall / horizontal / vertical accuracies
* `Intent` 评测（GT bbox）：同上，但跳过 bbox 匹配，直接对齐 GT 目标

---

### 5.2 Risk 评测：`risk_eval_qwen.py`

脚本位置：

```text
evaluation/risk_eval_qwen.py
```

功能：

* 加载 GT Risk (`Risk` 字段)：
  `../DRAMA-X_hf/drama_x_annotations_populated.jsonl`
* 加载预测 Risk：
  `../drama_intent/outputs/qwen3_local/qwen3_2b_pred_risk.json`
* 对齐 `sample_id` 后，计算：

  * Accuracy
  * Balanced Accuracy
  * Precision / Recall / F1（正类 = Risk=Yes）
  * 混淆矩阵 [[TN, FP], [FN, TP]]

运行：

```bash
python risk_eval_qwen.py
```

评测结果会同时：

* 打印在命令行；
* 保存到 `risk_eval_qwen3_2b.json`。

---

### 5.3 Action 评测：`action_eval_qwen.py`

脚本位置：

```text
evaluation/action_eval_qwen.py
```

功能：

1. 从 GT 中读取 `suggested_action` 文本；
2. 从预测中读取 `Suggested_action` 文本；
3. 使用简单规则将动作归一到 3 个类：

   * `STOP`：包含 stop / halt / yield 等；
   * `SLOW`：包含 slow / cautious / careful / aware 等；
   * `GO`  ：包含 continue / proceed / maintain / drive / follow 等；
   * GT 为 `N/A` 的样本不参与评测；
4. 计算：

   * STOP/SLOW/GO 的分类 Accuracy / Macro-F1 / Weighted-F1；
   * 混淆矩阵（行 = GT，列 = Pred）；
5. 如果安装了 `bert-score`，额外计算动作文本之间的 BERTScore（P / R / F1）。

运行：

```bash
python action_eval_qwen.py
```

评测结果同时：

* 打印在命令行；
* 保存到 `action_eval_qwen3_2b.json`。

---

## 6. Qwen3-VL-2B @ DRAMA-X 基线结果（TODO1 第一版）

### 6.1 Intent Prediction (IP) + BBox

**(A) End-to-end：预测 BBox + 预测 Intent**

* BBox：

  * overall accuracy ≈ **0.0059**（约 0.6%）
  * 总目标数：5750
  * 正确检测：34
* Intent（在预测框基础上）：

  * overall / horizontal / vertical accuracy 全部 ≈ **0.0**

> 结论：
>
> * Qwen3-VL-2B 作为“检测器 + 意图预测器”的端到端方案几乎不可用；
> * 检测这一步完全崩溃，导致 Intent 评测也被拖到 0；
> * **后续必须将 OD 从 VLM 中剥离出来，采用专门的检测模型或使用 GT bbox。**

**(B) 使用 GT BBox，只评 Intent**

* overall intent accuracy ≈ **0.242**
* horizontal accuracy ≈ **0.392**（左右方向）
* vertical accuracy ≈ **0.092**（towards / away / stationary 维度）

> 结论：
>
> * 在知道目标框位置的前提下，Qwen 在“向左/向右/静止”这类 **水平几何关系** 上还能学到一点信息；
> * 在“朝向自车 / 远离自车 / 静止”这类 **时序/速度相关** 信息上几乎无能为力（vertical ≈ 9%）；
> * **单帧输入 + 纯视觉提示无法支撑细粒度的 intent 预测，尤其是纵向意图。**

---

### 6.2 Risk Assessment (RA)

使用 `risk_eval_qwen.py` 的结果：

* GT 样本数：5686
* 有预测的样本数：4381（overlap 4379）

主要指标：

| 指标                                 | 数值                       |
| ---------------------------------- | ------------------------ |
| Accuracy                           | **0.384**                |
| Balanced Accuracy                  | **0.587**                |
| Precision (Risk=Yes)               | **0.986**                |
| Recall (Risk=Yes)                  | **0.373**                |
| F1 (Risk=Yes)                      | **0.541**                |
| Confusion matrix [[TN,FP],[FN,TP]] | [[89, 22], [2677, 1591]] |

> 解释：
>
> * 数据高度不平衡：在 overlap 的 4379 样本中，正类（Risk=Yes）约 4268 条，负类约 111 条；
> * Precision 高达 ~0.99：**一旦模型说 Risk=Yes，几乎都是对的**；
> * Recall 只有 ~0.37：**漏检大量 Risk=Yes 的样本**（FN = 2677）；
> * Balanced Accuracy ~0.59，略高于“永远预测 Yes” 的 0.5 baseline，但仍然偏低。

> 结论：
>
> * 当前的 Risk 输出 **非常保守**：宁可错过大量风险，也要保证一旦判 Risk=Yes 就几乎正确；
> * 这种行为更像“慢系统高门槛告警”，不适合作为高召回的实时安全指标；
> * 后续设计快慢系统时：
>
>   * 快系统（risk score / indicator）必须优先保证 recall；
>   * 慢系统（CoT reasoning）可以偏精度，作为解释与辅助决策。

---

### 6.3 Action Suggestion (AS)

使用 `action_eval_qwen.py` 的结果：

* 用于评测的样本数：4266（GT=NA 的样本和缺预测的样本被丢弃）

类别分布（GT）：

* GO： 388
* SLOW：2501
* STOP：1377

1）**三分类（STOP / SLOW / GO）指标：**

| 指标          | 数值         |
| ----------- | ---------- |
| Accuracy    | **0.2567** |
| Macro F1    | **0.2715** |
| Weighted F1 | **0.2661** |

混淆矩阵（行 = GT，列 = Pred；顺序：GO, SLOW, STOP）：

```text
[[ 205,  37, 146],   # GT = GO
 [2009, 178, 314],   # GT = SLOW
 [ 560, 105, 712]]   # GT = STOP
```

可见：

* GT = GO 的 recall ≈ 205 / 388 ≈ **52.8%**
* GT = STOP 的 recall ≈ 712 / 1377 ≈ **51.7%**
* GT = SLOW 的 recall ≈ 178 / 2501 ≈ **7.1%**（几乎全被误判为 GO）

> 结论：
>
> * Qwen3-VL-2B 在动作层面 **严重低估“需要谨慎减速/绕行”的中风险场景**；
> * 大部分 SLOW 场景被预测为 GO，是一个明显的安全隐患。

2）**BERTScore（文本级相似度）：**

| 指标             | 数值         |
| -------------- | ---------- |
| precision_mean | **0.0119** |
| recall_mean    | **0.0477** |
| f1_mean        | **0.0302** |

> 文本动作建议与 GT 文本的语义相似度非常低，说明：
>
> * Qwen 给出的动作建议多为“泛安全建议”，与具体场景关联度不强；
> * 当前的 CoT / prompt 没有有效地约束动作为结构化、可评测的形式。

---

## 7. 小结：TODO1 完成情况 & 后续方向

### 7.1 TODO1 完成情况

* ✅ **Intent Baseline**：

  * 端到端（BBox+Intent）→ 几乎为 0；
  * GT BBox → overall ≈ 24%，horizontal ≈ 39%，vertical ≈ 9%。
* ✅ **Risk Baseline**：

  * 高 precision（≈0.99），低 recall（≈0.37），balanced acc ≈ 0.59。
* ✅ **Action Baseline**：

  * STOP/SLOW/GO Accuracy ≈ 0.26，Macro F1 ≈ 0.27；
  * SLOW 类几乎被全判成 GO；
  * 文本 BERTScore F1 ≈ 0.03。

综上，**TODO1（DRAMA-X + Qwen3-VL-2B 的三任务 zero-shot baseline）已全部完成**，并对各自 failure mode 有了清晰认识。

### 7.2 对后续研究/项目的启示（简要）

1. **检测能力必须从 VLM 中剥离**

   * 端到端由 VLM 自己报 bbox 完全不可行，必须使用专门 OD（或 GT bbox）作为输入条件。

2. **Intent 垂直维度需要显式时序/轨迹信息**

   * 单帧 + 文本提示无法解决“towards/away”问题，后续 CoC / 数据设计要显式给出时间序列 / 相对运动。

3. **Risk 输出要区分快/慢系统职责**

   * 快系统：高召回、实时、数值化的 risk indicator（TTC / DTC / relative distance 等）；
   * 慢系统：基于更丰富上下文的 CoC/CoT 推理，用于解释和补充，而不是唯一判定依据。

4. **Action 建议需要强结构化和因果约束**

   * 目前 Qwen 给的自由文本与 GT 差异巨大，难以评测与对齐；
   * 未来需要借鉴 Alpamayo-R1 等工作，把动作和理由放进统一的 CoC 模板中（结构化输出），并与风险指标对齐。

后续的工作（TODO2+）将基于这套 baseline，设计更合理的 **CoC-style 标注与微调流程**，实现：

* 更可靠的 risk indicator；
* 可解释且与安全相关的 reasoning；
* 内外信息联动的快慢双系统。

```

::contentReference[oaicite:0]{index=0}
```
