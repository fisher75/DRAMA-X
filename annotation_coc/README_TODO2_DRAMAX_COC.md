# DRAMA-X TODO2: Chain-of-Causation (CoC) Annotation Generation

## 1. 背景与目标

### 1.1 TODO2 是什么？
TODO2 的目标是：**为 DRAMA-X 数据集中的每一个样本生成一条符合 CoC（Chain-of-Causation）风格的高质量推理标注**。  
该风格参考 **NVIDIA R1 系列论文中的 CoC / causal reasoning 写法**，强调：

- 因果清晰（Cause → Effect）
- 决策可解释（Explainable decision making）
- 推理短而精炼（1–3 句）
- 不允许凭空捏造（No hallucination）

最终输出将被用于：
- 可解释驾驶决策建模
- 后续 VLM / VLA 训练或评测
- DRAMA-X 在 reasoning 维度的增强版本

---

## 2. CoC 标注定义（最终输出规范）

每个样本必须 **且只能** 生成如下 JSON 结构：

```json
{
  "driving_decision": "GO | SLOW | STOP",
  "critical_components": {
    "primary_vru": {
      "id": "vru_1",
      "type": "pedestrian",
      "intent": "crossing",
      "distance": "near",
      "position": "front-left"
    },
    "secondary_context": {
      "traffic_density": "low",
      "visibility": "clear"
    }
  },
  "coc_trace": "Because the pedestrian is crossing close to the ego vehicle, therefore the ego vehicle should slow down to avoid potential collision."
}
```
---

# DRAMA-X TODO2：CoC 风格标注生成（Qwen3-VL / NVIDIA R1 CoC 风格）

## 目标（TODO2）

为 DRAMA-X 的每条样本（图片 + structured facts）生成一段 **Chain-of-Causation (CoC)** 风格的推理标注，产出字段严格为：

```json
{
  "driving_decision": "GO|SLOW|STOP",
  "critical_components": { ... },
  "coc_trace": "Because ... therefore ..."
}
```

并将其与原 structured facts 合并，写成新的 JSONL（可用于后续训练/评测/分析）。

> 这个 TODO2 的核心思想：参考 “NVIDIA R1 CoC 风格”——强调**因果链条**、短而清晰、可解释，不允许凭空捏造。

---

## 文件与输入输出

### 输入：`drama_x_structured.jsonl`

由 `build_structured_facts.py` 生成，每行一个样本，包含（至少）：

* `sample_id`（或 `id`）
* `image_path`（本地路径或 http/https URL）
* `risk_label` / `Risk`
* `suggested_action_raw` / `Suggested_action`
* `vru_list`：VRU 列表（如行人/骑行者等），含属性：

  * `vru_id`, `type`, `intent_list`, `position_category`, `distance_level`, `description` 等

### 输出：`drama_x_coc_qwen3vl_2b*.jsonl`

每行一个样本：在原 structured fields 基础上，合并写入：

* `driving_decision`
* `critical_components`
* `coc_trace`

若使用 `--preserve_order`，会在数据并行中临时写入 `_line_index`，最终 merge 时会按原输入行号恢复顺序并删除 `_line_index`。

---

## 方法概述：用 Qwen3-VL 自动生成 CoC

脚本：`build_coc_with_qwen.py`

### 推理流程

对每条样本执行：

1. 读入 `image_path` 加载图片（支持本地/URL）
2. 用 structured facts 构造 prompt（包含 VRU 列表、risk label、suggested action 等）
3. 用 `processor.apply_chat_template` 构造多模态对话输入（image + text）
4. `processor(text=[...], images=[...])` 得到张量输入
5. **关键：保证 `pixel_values` 与 `image_grid_thw` 一致**

   * 将 `pixel_values` 规范为 `(B, N, C)`（有些版本会给 `(N, C)`）
   * 将 `image_grid_thw` 规范为 `(B, 3)`
   * 强制检查 `t*h*w == N`，否则根据 N 分解因子 + 原图宽高比，自动修正 grid
6. 先做一次 forward self-check（可选 debug），通过后再 `model.generate`
7. decode 输出并从文本中抽取第一个 “平衡括号 JSON”（更稳）

---

## 运行方式

### 1）推荐：数据并行（Data Parallel）

特点：每张卡各加载一份模型，处理不同切片样本，**通常比 device_map 模型并行更快更稳**（尤其是 2B + 多样本推理）。

#### 小规模测试（强烈建议先跑）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python build_coc_with_qwen.py \
  --structured_path /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_structured.jsonl \
  --output_path /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b_debug.jsonl \
  --model_path /workspace/models/VLM/Qwen3-VL-2B-Instruct \
  --data_parallel \
  --max_samples 20 \
  --preserve_order \
  --verbose_debug
```

#### 全量运行

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python build_coc_with_qwen.py \
  --structured_path /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_structured.jsonl \
  --output_path /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b.jsonl \
  --model_path /workspace/models/VLM/Qwen3-VL-2B-Instruct \
  --data_parallel \
  --preserve_order
```

> 说明：
>
> * `--preserve_order` 会确保最终输出顺序与输入 JSONL 行顺序一致（对后处理/对齐很重要）。
> * `--max_samples` 不指定或为 -1 表示全量。

---

### 2）单进程模型并行（Model Parallel / device_map="auto"）

特点：模型会被切分到可见 GPU 上（适合显存不够、或只想快速验证一张卡/两张卡）。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python build_coc_with_qwen.py \
  --structured_path /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_structured.jsonl \
  --output_path /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b_mp.jsonl \
  --model_path /workspace/models/VLM/Qwen3-VL-2B-Instruct \
  --max_samples 20 \
  --verbose_debug
```

---

## 输出质量：为什么建议跑“全量”？

从原理上讲：

* CoC 标注是 **对每条样本独立生成**，不存在训练那种“全量会让模型更好”的概念。
* “跑全量更好”的意义在于：

  1. 你最终要用 CoC 标注覆盖全数据集（用于训练/评测/统计）
  2. 能发现 edge cases（异常图片、异常 grid、缺失字段、极端 VRU 组合）
  3. 更完整的可解释性分析、覆盖更多风险场景

因此推荐做法：

* **先小样本 debug（10~50 条）确保稳**
* 然后全量跑（并保留日志/异常样本列表）

---

## 常见坑与排查（重要）

### A. `image_grid_thw` / `pixel_values` 不一致导致 Qwen3-VL 内部 split 崩溃

典型报错（历史出现过）：

* `split_with_sizes expects split_sizes have only non-negative entries`
* 或各种 attention 内部 split lengths 异常

本项目的解决思路：

* **永远保证 `t*h*w == N`**

  * N = `pixel_values` token 数
  * `image_grid_thw = [t, h, w]`
* 代码里有硬检查日志：

  * `[CHECK] pixel_values N = ...`
  * `[CHECK] image_grid_thw = [t,h,w] product = ...`
* 如发现 mismatch，会自动根据 N 的因子与原图宽高比修复 `(h,w)`

> 这一步是 TODO2 能稳定全量跑通的关键点之一。

---

### B. tmux 里 “明明显示已激活 conda env，但 which python 不对”，导致 `No module named torch`

现象：

* prompt 前缀显示 `(/workspace/conda_envs/dramax)`
* 但 `which python` 却是 `/opt/anaconda3/bin/python`
* 进而出现 `ModuleNotFoundError: No module named 'torch'`

原因（常见）：

* tmux session 里 PATH/conda 初始化没有正确加载
* 或你在某个 shell 初始化脚本里覆盖了 PATH
* 或你开 tmux 前后环境不同，shell 没 reload

解决建议（强烈建议统一执行方式）：

1. **进入 tmux 后重新激活环境**

```bash
conda deactivate || true
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /workspace/conda_envs/dramax
which python
python -c "import torch; print(torch.__version__)"
```

2. 或者最稳的：**直接用环境里的 python 跑脚本**

```bash
/workspace/conda_envs/dramax/bin/python build_coc_with_qwen.py ...
```

3. 若你用的是 `conda activate dramax`（名字），确保该 env 的 prefix 真的是 `/workspace/conda_envs/dramax`。

---

### C. 如何查看 GPU 占用 / 进程是谁

```bash
nvidia-smi
```

更详细（含命令行）：

```bash
nvidia-smi pmon -c 1
```

查某个 PID 在跑什么：

```bash
ps -fp <PID>
```

如果发现有另一组 python（例如别的 env）也在占卡，会显著拖慢全量生成速度，应先确认是否需要并考虑清理。

---

## 输出格式与约束（对齐 CoC 风格）

输出必须：

* 只有 3 个 key：`driving_decision`, `critical_components`, `coc_trace`
* `driving_decision` 严格是 `GO|SLOW|STOP` 之一
* `coc_trace` 1–3 句，**Because … therefore …** 风格
* **不允许捏造**（只能依据图像 + structured facts 推断）

---

## 建议的协作规范

* 全量跑之前：先 `--max_samples 20 --verbose_debug` 确认无异常
* 全量跑：建议写日志（可以用 `tee`）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python build_coc_with_qwen.py ... --data_parallel --preserve_order \
|& tee run_coc_full.log
```

* 若中途失败：记录失败 sample_id / 行号，后续可做重跑或过滤

---

## 最终产物（TODO2 Done 标志）

当你拿到：

* `drama_x_coc_qwen3vl_2b.jsonl`（全量）
* 样本格式一致、顺序可控、无大量缺失
* 随机抽查若干条 CoC 合理、无明显幻觉

则 TODO2 完成。

---

