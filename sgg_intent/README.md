# 📘 DRAMA-X 本地推理（Qwen3-VL 本地版）README

（适用于 `/workspace/chz/code/DRAMA-X/` 项目结构）

---

# 1. 数据集准备

需要两个来源的数据：
① 来自 HuggingFace 的 **drama_x_annotated.jsonl**（标注文件）
② 来自 GitHub / 官方 ZIP 解包的 **integrated_output_v2.json**（包含 image_path 和 video_path）

最终我们会把它们合并成一个 **统一的 JSONL**，再进一步转成推理脚本需要的 JSON 字典格式。

---

## 1.1 下载 DRAMA-X（HuggingFace）

到 HuggingFace 页面：
[https://huggingface.co/datasets/mgod96/DRAMA-X](https://huggingface.co/datasets/mgod96/DRAMA-X)

下载：

```
drama_x_annotated.jsonl
```

存放到工程路径（示例）：

```
/workspace/chz/code/DRAMA-X/DRAMA-X_hf/drama_x_annotated.jsonl
```

---

## 1.2 下载 integrated_output_v2.json（官方 ZIP）

从官方提供的链接下载 ZIP，解压后获得：

```
integrated_output_v2.json
```

放到同一目录：

```
/workspace/chz/code/DRAMA-X/DRAMA-X_hf/integrated_output_v2.json
```

---

# 2. 填充 image_path / video_path

使用官方的 populate 脚本：

```
python DRAMA-X_hf/populate_drama_x.py \
  DRAMA-X_hf/drama_x_annotated.jsonl \
  DRAMA-X_hf/integrated_output_v2.json \
  -o DRAMA-X_hf/drama_x_annotations_populated.jsonl
```

输出文件：

```
DRAMA-X_hf/drama_x_annotations_populated.jsonl
```

这个文件已经包含了：

* image_path（完整 URL 或本地路径）
* video_path
* 所有需要的 annotation

---

# 3. 转换为 updated_output.json（pipeline 标准格式）

SGG-Intent 代码需要的是 **dict keyed by id** 的 JSON，而不是 JSONL、不是 list。

运行以下脚本：

```bash
cd /workspace/chz/code/DRAMA-X

python - << 'PY'
import json, os

root = "/workspace/chz/code/DRAMA-X"
hf_dir = os.path.join(root, "DRAMA-X_hf")
in_path = os.path.join(hf_dir, "drama_x_annotations_populated.jsonl")

out_dir = os.path.join(root, "drama_intent")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "updated_output.json")

data = {}
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        fid = str(rec.get("id"))
        data[fid] = rec

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print("Wrote", out_path, "with", len(data), "records")
PY
```

将得到：

```
/workspace/chz/code/DRAMA-X/drama_intent/updated_output.json
```

结构示例：

```json
{
  "clip_305_000786_frame_000786": {
      "id": "clip_305_000786_frame_000786",
      "image_path": "...",
      "video_path": "...",
      ...
  },
  "clip_1111_000552_frame_000552": { ... }
}
```

这是 Qwen3 推理代码唯一认可的输入格式。

---

# 4. 配置模型路径（2B / 8B 都可以）

编辑文件：

```
/workspace/chz/code/DRAMA-X/sgg_intent/qwen3_local_sgg_intent.py
```

找到：

```python
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN3_VL_MODEL_PATH",
    "/workspace/models/VLM/Qwen3-VL-2B-Instruct",
)
```

改成模型路径，例如：

```
/workspace/models/VLM/Qwen3-VL-8B-Instruct
```

或者继续用 2B：

```
/workspace/models/VLM/Qwen3-VL-2B-Instruct
```

---

# 5. **单 GPU** 推理（调试用）

先确认 updated_output.json 正确：

```
python sgg_intent/qwen3_local_sgg_intent.py --start 0 --end 4 --raw_mode 1
```

其中：

* `--raw_mode 1` = 一阶段（Risk + Suggested_action + Intent）
* `--raw_mode 0` = 两阶段（Scene Graph + Intent）

输出在：

```
drama_intent/outputs/qwen3_local/
```

---

# 6. **多 GPU 并行推理**（正式跑）

直接运行：

```bash
cd /workspace/chz/code/DRAMA-X/sgg_intent
python run_multi_gpu.py
```

在 `run_multi_gpu.py` 顶部可以设置：

```python
NUM_GPUS = 2       # 使用几张卡
RAW_MODE = True    # True=一阶段；False=两阶段
```

✓ **只改这一处即可**
不要在主推理文件里改 `raw=True/False`。

---

# 7. 输出文件结构

一阶段（RAW_MODE=True）输出：

```
all_raw_Qwen3-VL-2B-Instruct_onepass_gpu0.json
all_raw_Qwen3-VL-2B-Instruct_onepass_gpu1.json
all_raw_Qwen3-VL-2B-Instruct_onepass.json   (合并后的)
```

内容示例：

```json
{
  "clip_305_000786_frame_000786": {
    "Risk": "No",
    "Suggested_action": "...",
    "pedestrian_1": {
      "Intent": ["goes to the right", "moves away from ego vehicle"],
      "Reason": "...",
      "Bounding_box": [...]
    },
    ...
  },
  ...
}
```

两阶段（RAW_MODE=False）输出：

```
all_scene_graphs_..._gpu0.json
all_scene_graphs_..._gpu1.json
all_scene_graphs_...json

all_intent_jsons_..._gpu0.json
all_intent_jsons_..._gpu1.json
all_intent_jsons_...json
```

---

# 8. 工程目录结构（最终理想结构）

```
DRAMA-X/
│
├── DRAMA-X_hf/
│   ├── drama_x_annotated.jsonl
│   ├── integrated_output_v2.json
│   └── drama_x_annotations_populated.jsonl
│
├── drama_intent/
│   ├── updated_output.json   ← 实际推理的唯一入口文件
│   └── outputs/
│       └── qwen3_local/
│           ├── all_raw_..._gpu0.json
│           ├── all_raw_..._gpu1.json
│           └── all_raw_....json
│
├── sgg_intent/
│   ├── qwen3_local_sgg_intent.py
│   └── run_multi_gpu.py
│
└── (其他 DRAMA-X 代码)
```

---

# 9. 最终要记住的“黄金三步”

### **① 用 populate 脚本处理 HF 文件（得到 populated.jsonl）**

### **② 转为 updated_output.json（统一 dict 格式）**

### **③ 用 run_multi_gpu.py + RAW_MODE = True 跑 Qwen3 一阶段推理**

这就是现在的完整 pipeline。
