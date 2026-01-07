# DRAMA-X TODO2: Chain-of-Causation (CoC) Annotation Generation Pipeline (v2)

**Version:** 2.0 (Updated 2026-01-08)  
**Task:** External Risk Reasoning / CoC Label Generation  
**Model:** Qwen3-VL-2B-Instruct (Local Deployment)

---

## 1. 背景与目标 (Background)

### 1.1 核心任务
本项目标（TODO2）旨在为 **DRAMA-X** 数据集中的每一个高危样本生成一条符合 **CoC (Chain-of-Causation)** 风格的高质量推理标注。这些标注将用于后续训练一个具备“可解释性”和“物理一致性”的自动驾驶风险推理模型。

### 1.2 v2 版本关键更新 (Key Updates)
相比早期版本，v2 流程引入了以下核心改进：

1.  **Risk-based Primary VRU Selection (基于风险的主目标选择)**:
    * **旧策略**: 简单选取 BBox 面积最大的 VRU（容易选中路边无关的大目标）。
    * **新策略 (v2)**: 引入基于规则的 **Risk Score**，综合考量 **位置 (Position)**、**意图 (Intent)**、**距离 (Distance)** 和 **尺寸 (Size)**。
    * **收益**: 确保生成的 CoC 聚焦于真正导致危险的目标（如横穿马路的行人），而非视觉上最显眼的目标。

2.  **Robust Vision Pipeline (视觉输入修复)**:
    * 修复了 Qwen3-VL 在处理部分图像时出现的 `split_with_sizes=[0]` 及 `CUDA error`。
    * 引入 `qwen_vl_utils` 标准预处理流程和 `load_image_robust` 机制。

3.  **Efficiency (效率优化)**:
    * 全面支持 **数据并行 (Data Parallel)**，在 4x A5000 环境下吞吐量提升 4 倍。

---

## 2. 方法论：Risk Score 策略详解

为了选出最关键的风险目标 (`primary_vru`)，我们在 `build_structured_facts_v2.py` 中实现了如下打分逻辑：

$$Score(v) = w_p \cdot \text{Pos}(v) + w_i \cdot \text{Intent}(v) + w_d \cdot \text{Dist}(v) + w_s \cdot \text{Size}(v)$$

### 权重设计 (Weights Rationale)

* **$w_p = 0.45$ (Position)**: **最高权重**。依据 RSS (Responsibility-Sensitive Safety) 理念，只有位于自车行驶路径（Front/Ego-Lane）上的对象才构成直接冲突风险。
* **$w_i = 0.35$ (Intent)**: **次高权重**。反映对象的动态趋势。`Crossing` / `Moving toward` (横穿/靠近) 的风险远高于 `Standing` (静止) 或 `Moving away` (远离)。
* **$w_d = 0.15$ (Distance)**: 距离越近风险越大，作为 TTC (Time-to-Collision) 的静态代理。
* **$w_s = 0.05$ (Size)**: **弱特征**。仅作为 Tie-breaker，防止因距离误判而忽略大目标。

---

## 3. 流水线与脚本 (Pipeline Structure)

整体流程分为两步：**结构化事实构建** -> **大模型推理生成**。

### Step 1: 结构化事实生成 (CPU)
* **脚本**: `build_structured_facts_v2.py`
* **输入**: 原始 DRAMA-X 标注 (`drama_x_annotations_populated.jsonl`)
* **逻辑**: 
    1. 解析原始 JSON。
    2. 对每个样本内的所有 VRU 计算 **Risk Score**。
    3. 按分数降序排列 VRU 列表（`vru_list[0]` 即为 Primary）。
    4. 写入新的 v2 版中间文件。
* **输出**: `drama_x_structured_v2.jsonl`

### Step 2: CoC 推理生成 (GPU)
* **脚本**: `build_coc_with_qwen.py`
* **输入**: `drama_x_structured_v2.jsonl`
* **模型**: Qwen3-VL-2B-Instruct (本地加载)
* **逻辑**:
    1. 读取 v2 版数据，构建 Prompt（将 Primary VRU 信息置顶）。
    2. 加载图片（使用 Robust Loader）。
    3. Qwen3-VL 推理，生成 JSON 格式的 CoC。
    4. 提取并校验 JSON，与原数据合并。
* **输出**: `drama_x_coc_qwen3vl_2b_v2_full.jsonl`

---

## 4. 执行指南 (Execution Guide)

### 环境准备
确保已安装最新依赖，特别是 Qwen 的视觉工具：
```bash
pip install qwen-vl-utils

```

### 第一步：生成带 Risk 排序的结构化数据

```bash
cd /workspace/chz/code/DRAMA-X/annotation_coc

# 注意替换 --input 为你实际的原始数据路径
python build_structured_facts_v2.py \
  --input ../DRAMA-X_hf/drama_x_annotations_populated.jsonl \
  --output drama_x_structured_v2.jsonl

```

### 第二步：运行 Qwen3-VL 生成 CoC

#### 选项 A：小规模 Debug (推荐先跑这个)

先跑 20 条，检查 `critical_components` 里的 Primary VRU 是否合理。

```bash
export CUDA_VISIBLE_DEVICES=0,1
python build_coc_with_qwen.py \
  --structured_path drama_x_structured_v2.jsonl \
  --output_path drama_x_coc_qwen3vl_2b_v2_debug.jsonl \
  --model_path /workspace/models/VLM/Qwen3-VL-2B-Instruct \
  --max_samples 20 \
  --verbose_debug

```

#### 选项 B：全量生成 (数据并行)

确认无误后，使用 4 卡并行跑全量。建议使用 `nohup`。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python build_coc_with_qwen.py \
  --structured_path drama_x_structured_v2.jsonl \
  --output_path drama_x_coc_qwen3vl_2b_v2_full.jsonl \
  --model_path /workspace/models/VLM/Qwen3-VL-2B-Instruct \
  --data_parallel \
  --preserve_order \
  > run_coc_v2_full.log 2>&1 &

# 查看日志
tail -f run_coc_v2_full.log

```

---

## 5. 输出规范 (Output Schema)

最终生成的 JSONL 文件中，每行样本将包含以下核心字段：

```json
{
  "sample_id": "clip_...",
  "risk_label": "high",
  "vru_list": [ ... ], // 已按 Risk Score 排序
  
  // --- 模型生成的 CoC 部分 ---
  "driving_decision": "SLOW",  // 严格枚举: GO | SLOW | STOP
  "critical_components": {
    "primary_vru": {
      "id": "ped_1",           // 对应 vru_list[0]
      "type": "pedestrian",
      "intent": "crossing",
      "position": "front-center"
    }
  },
  "coc_trace": "Because the pedestrian is crossing the road directly in front of the ego vehicle (high risk), therefore the ego vehicle should slow down to prepare for a stop."
}

```

---

## 6. 常见问题排查 (Troubleshooting)

* **报错**: `RuntimeError: split_with_sizes ... got split_sizes=[0]`
* **原因**: 图片加载失败或尺寸异常，导致 Qwen 视觉编码器计算 Grid 错误。
* **解决**: 脚本已内置 `load_image_robust` 和 `process_vision_info` 修复此问题。如仍出现，请检查 Log 中报错的具体图片路径是否损坏或 404。


* **报错**: `CUDA error: device-side assert triggered`
* **原因**: 通常紧随上面的 `split_sizes` 错误发生，导致 CUDA 上下文损坏。
* **解决**: **必须重启 Python 进程**（如果是 Notebook 需重启 Kernel）。检查输入数据完整性。


* **警告**: `[WARN] JSON parse failed`
* **原因**: 模型生成的文本没有严格遵循 JSON 格式。
* **解决**: 脚本内置了 `extract_first_balanced_json` 和基础修复逻辑。少量失败（<1%）可忽略，若失败率高，需微调 Prompt。



```

```