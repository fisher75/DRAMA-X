#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本脚本：本地 Qwen3-VL 版 SGG-Intent 推理
路径建议：/workspace/chz/code/DRAMA-X/sgg_intent/qwen3_local_sgg_intent.py
"""

import os
import json
import time
import re
from io import BytesIO
from typing import Dict, Any
import argparse

import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# 尝试导入 openai（用于可选的 JSON 修复 fallback，没有则静默跳过）
try:
    import openai  # type: ignore
except ImportError:
    openai = None

# ========= 命令行参数 =========
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--gpu", type=int, default=0)       # 只是记录用，不在这里绑 GPU
parser.add_argument("--raw_mode", type=int, default=1)  # 1=one-pass, 0=two-stage
args = parser.parse_args()

# 来自启动脚本的 raw 模式：True=一阶段，False=两阶段
RAW_MODE_FROM_LAUNCHER = bool(args.raw_mode)

# ====== 默认路径配置（请根据你的环境修改） ======
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

# 默认数据集 JSON 路径
DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "drama_intent", "updated_output.json")

# 默认模型路径（请改成你实际存放 Qwen3-VL 的本地目录）
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN3_VL_MODEL_PATH",
    "/workspace/models/VLM/Qwen3-VL-2B-Instruct",
)

def load_dataset_dict(path: str) -> Dict[str, Any]:
    """
    同时兼容两种情况：
    1) 整个文件是一个 JSON 对象（dict 或 list）
    2) JSON Lines：每一行一个 JSON（HuggingFace Dataset.to_json 默认写法）

    返回形式统一为：Dict[str, Any]
      - 如果原本是 dict：直接返回
      - 如果是 list 或 JSONL：按 0,1,2,... 这些字符串做 key
    """
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        return {}

    # 先尝试按“单个大 JSON 对象”解析
    try:
        obj = json.loads(text)
        # 如果是 list，也统一转成 {"0": item0, "1": item1, ...}
        if isinstance(obj, list):
            return {str(i): rec for i, rec in enumerate(obj)}
        elif isinstance(obj, dict):
            return obj
        else:
            # 其它类型（比如标量），就也转一下
            return {"0": obj}
    except json.JSONDecodeError:
        pass

    # 如果直接 json.loads 失败，再试 JSON Lines（每行一个 JSON）
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    return {str(i): rec for i, rec in enumerate(records)}

class Qwen3LocalSGGInference:
    """
    本地 Qwen3-VL 版本的 SGG-Intent 推理器：
    - 不依赖 HTTP 接口
    - 完全在本地 GPU 上推理
    """

    def __init__(
        self,
        dataset_path: str,
        model_path: str,
        max_tokens: int = 512,
        rate_limit: int = 1,       # 本地推理默认不开太多并发，防止显存爆
        http_timeout: int = 60,    # 仅用于下载远程图片时
        max_retries: int = 2,
    ):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.rate_limit = rate_limit
        self.http_timeout = http_timeout
        self.max_retries = max_retries

        # 加载本地 Qwen3-VL 模型与 Processor
        print(f"[INFO] Loading Qwen3-VL model from: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.device = self.model.device
        print(f"[INFO] Model loaded on device: {self.device}")

        # Prompt 模板
        self.scene_graph_prompt_template = self._scene_graph_prompt()
        self.intent_prompt_template = self._intent_prompt()
        self.all_gen_prompt_template = self._all_gen_prompt()

        # 可选：用于 JSON fallback 的 OpenAI key（若不存在则不启用）
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai is not None and self.openai_api_key:
            openai.api_key = self.openai_api_key
            print("[INFO] OPENAI_API_KEY detected, JSON fallback via gpt-4o-mini ENABLED.")
        else:
            print("[INFO] OPENAI_API_KEY not found or openai package missing, "
                  "JSON fallback via gpt-4o-mini DISABLED.")

    def _build_tag(self, raw: bool) -> str:
        """
        根据当前模型路径和推理模式生成一个 tag，用于区分输出文件名。
        例如：
        - /workspace/models/VLM/Qwen3-VL-2B-Instruct + raw=True
          -> Qwen3-VL-2B-Instruct_onepass
        """
        model_tag = os.path.basename(self.model_path.rstrip("/"))
        mode_tag = "onepass" if raw else "twostage"
        return f"{model_tag}_{mode_tag}"

    # ====== 基础函数 ======

    def load_data(self) -> Dict[str, Any]:
        return load_dataset_dict(self.dataset_path)


    def _scene_graph_prompt(self) -> str:
        scene_graph_prompt = """
For the provided image, generate a scene graph in JSON format that includes the following, be concise and consider only important objects:
1. Objects in the frame. The special requirement is that you must include every pedestrian and cyclist separately and not group them as people or cyclists.
2. Object attributes inside object dictionary that are relevant to answering the question. Object attributes should include the state of the object e.g., moving or static, description of the object such as color, orientation, etc.
3. Object bounding boxes. These should be with respect to the original image dimensions [x1, y1, x2, y2].
4. Object relationships between objects. This should be detailed, up to 4 words.

Limit your response to at most 5 most relevant objects in the scene.

An example structure would look like this:
{
  "Objects": {
    "name_of_object": {
      "attributes": [],
      "bounding_box": [x1, y1, x2, y2]
    }
  },
  "Relationships": [
    {
      "from": "name_of_object1",
      "to": "name_of_object2",
      "relationship": "relationship between obj_1 and obj_2"
    }
  ]
}

Strictly output ONLY ONE valid JSON object. Do NOT output any explanation.
Scene Graph:
"""
        return scene_graph_prompt.strip()

    def _intent_prompt(self) -> str:
        intent_prompt = """
For the provided scene graph, image, and question, generate an object-intent JSON which includes the following:
1. All objects from the scene graph.
2. Predicted intent for every object. Intent should be one of these values:

2.1 Lateral (Sideways) Intent Options (exactly one of these):
     - "goes to the left"
     - "goes to the right"

2.2 Vertical Intent Options (exactly one of these):
     - "moves away from ego vehicle"
     - "moves towards ego vehicle"
     - "stationary"

3. Reason for this prediction (in natural language).
4. Bounding box of the object [x1, y1, x2, y2] with respect to original image dimensions.

The output MUST be a dictionary of the form:
{
  "name_of_object": {
    "Intent": ["predicted lateral intent", "predicted vertical intent"],
    "Reason": "reason for this prediction",
    "Bounding_box": [x1, y1, x2, y2]
  },
  ...
}

The "Intent" field list MUST ALWAYS have exactly two values (first lateral, then vertical).
Strictly output ONLY ONE valid JSON object. Do NOT output any explanation or scene graph details.
"""
        return intent_prompt.strip()

    def _all_gen_prompt(self) -> str:
        all_gen_prompt = """
For the provided image and question, generate an object-intent JSON which includes the following:
1. AT MOST 5 objects from the scene including pedestrians and cyclists.
2. Predicted intent for every object. Intent should be one of these values:

2.1 Lateral (Sideways) Intent Options (exactly one of these):
     - "goes to the left"
     - "goes to the right"

2.2 Vertical Intent Options (exactly one of these):
     - "moves away from ego vehicle"
     - "moves towards ego vehicle"
     - "stationary"

3. Risk score for this scene ("Yes" or "No"). Risk is defined as a hazardous scenario that poses danger to the ego vehicle.
4. Bounding box of each object [x1, y1, x2, y2] with respect to original image dimensions.
5. Suggested action for the ego vehicle given the scene and the risk score.

The output MUST be a single JSON object, for example:
{
  "Risk": "Yes" or "No",
  "Suggested_action": "suggested action for ego vehicle",
  "pedestrian_1": {
    "Intent": ["predicted lateral intent", "predicted vertical intent"],
    "Reason": "reason for this prediction",
    "Bounding_box": [x1, y1, x2, y2]
  },
  "car_1": {
    "Intent": ["predicted lateral intent", "predicted vertical intent"],
    "Reason": "reason for this prediction",
    "Bounding_box": [x1, y1, x2, y2]
  }
  ...
}

The "Intent" field list MUST ALWAYS have exactly two values (first lateral, then vertical).
Strictly output ONLY ONE valid JSON object. Do NOT output any explanation or scene graph details.
"""
        return all_gen_prompt.strip()

    # ====== 图像加载 + 推理封装 ======

    def _load_image(self, image_path: str) -> Image.Image:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            resp = requests.get(image_path, timeout=self.http_timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
        return img

    def _generate_with_image_and_text(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
            )

        input_ids = inputs["input_ids"]
        gen_ids = generated_ids[:, input_ids.shape[1]:]
        out_text = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return out_text.strip()

    # ====== JSON 提取+修复 ======

    def extract_and_fix_json(self, raw: str, prompt_type: str) -> dict:
        import json as pyjson

        def _extract_balanced(s: str) -> str:
            start = s.find("{")
            if start < 0:
                return s
            depth = 0
            for idx, ch in enumerate(s[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start: idx + 1]
            return s[start:]

        def _basic_fix(text: str) -> str:
            text = text.replace("“", '"').replace("”", '"')
            text = text.replace("‘", "'").replace("’", "'")
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = re.sub(r"\bTrue\b", "true", text)
            text = re.sub(r"\bFalse\b", "false", text)
            text = re.sub(r"\bNone\b", "null", text)
            text = re.sub(r"'(\w+)'\s*:", r'"\1":', text)
            text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
            text = re.sub(r",\s*(?=[}\]])", "", text)
            opens, closes = text.count("{"), text.count("}")
            if opens > closes:
                text += "}" * (opens - closes)
            elif closes > opens:
                text = "{" * (closes - opens) + text
            return text

        m = re.search(r"<BEGIN_JSON>(.*?)<END_JSON>", raw, re.S)
        fragment = m.group(1).strip() if m else raw.strip()

        candidate = _extract_balanced(fragment)

        try:
            return pyjson.loads(candidate)
        except pyjson.JSONDecodeError:
            pass

        fixed = _basic_fix(candidate)
        try:
            return pyjson.loads(fixed)
        except pyjson.JSONDecodeError:
            pass

        if openai is not None and self.openai_api_key:
            schema = (
                self.scene_graph_prompt_template
                if prompt_type == "scene"
                else self.intent_prompt_template
            )
            system = "You are a JSON formatter. Output only valid JSON."
            user = (
                "The following JSON is invalid. Your job is to only change "
                "the structural elements and not change the semantic content. "
                "Fix it to be valid JSON consistent with the following schema-like description:\n\n"
                f"{schema}\n\n"
                f"Broken JSON:\n```json\n{fixed}\n```"
            )
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
            corrected = resp.choices[0].message.content.strip()
            if corrected.startswith("```json"):
                corrected = corrected[7:]
            if corrected.endswith("```"):
                corrected = corrected[:-3]
            return pyjson.loads(corrected)
        else:
            raise pyjson.JSONDecodeError(
                "Failed to parse JSON after basic fixes and no JSON fallback available.",
                fixed,
                0,
            )

    # ====== 三个核心任务函数 ======

    def process_image_for_scene_graph(self, image_path: str) -> Dict[str, Any]:
        try:
            img = self._load_image(image_path)
            prompt = (
                f"{self.scene_graph_prompt_template}\n\n"
                "Analyse this image and output ONLY one JSON object."
            )
            out_text = self._generate_with_image_and_text(img, prompt)
            if out_text.startswith("```json"):
                out_text = out_text[7:]
            if out_text.endswith("```"):
                out_text = out_text[:-3]
            sg = self.extract_and_fix_json(out_text, prompt_type="scene")
            return sg
        except Exception as e:
            print(f"[ERROR] process_image_for_scene_graph failed for {image_path}: {e}")
            return {}

    def process_image_for_intent(
        self,
        image_path: str,
        scene_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            img = self._load_image(image_path)
            prompt = (
                f"{self.intent_prompt_template}\n\n"
                f"Scene graph (as JSON): {json.dumps(scene_graph, ensure_ascii=False)}\n\n"
                "Analyse this scene and output ONLY one valid JSON object."
            )
            out_text = self._generate_with_image_and_text(img, prompt)
            if out_text.startswith("```json"):
                out_text = out_text[7:]
            if out_text.endswith("```"):
                out_text = out_text[:-3]
            intent_json = self.extract_and_fix_json(out_text, prompt_type="intent")
            return intent_json
        except Exception as e:
            print(f"[ERROR] process_image_for_intent failed for {image_path}: {e}")
            return {}

    def process_image_one_pass(self, image_path: str) -> Dict[str, Any]:
        try:
            img = self._load_image(image_path)
            prompt = (
                f"{self.all_gen_prompt_template}\n\n"
                "Analyse this image and output ONLY one valid JSON object."
            )
            out_text = self._generate_with_image_and_text(img, prompt)
            if out_text.startswith("```json"):
                out_text = out_text[7:]
            if out_text.endswith("```"):
                out_text = out_text[:-3]
            intent_json = self.extract_and_fix_json(out_text, prompt_type="intent")
            return intent_json
        except Exception as e:
            print(f"[ERROR] process_image_one_pass failed for {image_path}: {e}")
            return {}

    # ====== 单帧处理 & 批量推理 ======

    def _process_frame(self, frame_id: str, frame_data: Dict[str, Any], raw: bool = False):
        try:
            image_path = frame_data["image_path"]
            if raw:
                combined_res = self.process_image_one_pass(image_path)
                return frame_id, combined_res, None
            else:
                sg = self.process_image_for_scene_graph(image_path)
                intent = self.process_image_for_intent(image_path, sg)
                return frame_id, sg, intent
        except Exception as e:
            print(f"[ERROR] Error processing frame {frame_id}: {e}")
            return frame_id, {}, {}

    def run_inference_dict(self, data_dict, overwrite=True, raw=True):
        """
        多GPU版本使用的核心函数：
        - 输入是子数据字典 data_dict
        - 每个进程写入独立的 *_gpuX.json，防止冲突
        """
        out_dir = os.path.join(
            os.path.dirname(self.dataset_path),
            "outputs",
            "qwen3_local"
        )
        os.makedirs(out_dir, exist_ok=True)

        # 标记当前模式 + 模型，用于文件名
        tag = self._build_tag(raw)  # 例如 Qwen3-VL-2B-Instruct_onepass

        # 子任务名称（避免 GPU 进程之间写文件冲突）
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        suffix = f"_gpu{gpu_id}"

        items = list(data_dict.items())
        print(f"[INFO] GPU {gpu_id} processing {len(items)} items")

        if raw:
            raw_path = os.path.join(
                out_dir,
                f"all_raw_{tag}{suffix}.json"
            )
            if overwrite or (not os.path.exists(raw_path)):
                all_raw = {}
            else:
                with open(raw_path, "r") as f:
                    all_raw = json.load(f)
                    items = [(fid, fdata) for fid, fdata in items if fid not in all_raw]
        else:
            sg_path = os.path.join(
                out_dir,
                f"all_scene_graphs_{tag}{suffix}.json"
            )
            intent_path = os.path.join(
                out_dir,
                f"all_intent_jsons_{tag}{suffix}.json"
            )

            if overwrite or (not os.path.exists(sg_path)):
                all_sg = {}
            else:
                with open(sg_path, "r") as f:
                    all_sg = json.load(f)

            if overwrite or (not os.path.exists(intent_path)):
                all_intent = {}
            else:
                with open(intent_path, "r") as f:
                    all_intent = json.load(f)

            items = [
                (fid, fdata)
                for fid, fdata in items
                if fid not in all_sg or fid not in all_intent
            ]

        print(f"[INFO] GPU {gpu_id} remaining items: {len(items)}")

        concur = max(1, self.rate_limit)
        chunks = [items[i: i + concur] for i in range(0, len(items), concur)]

        processed = 0
        with ThreadPoolExecutor(max_workers=concur) as exe:
            for batch in tqdm(chunks, desc=f"GPU {gpu_id} Processing"):
                futures = {
                    exe.submit(self._process_frame, fid, fdata, raw): fid
                    for fid, fdata in batch
                }

                for fut in as_completed(futures):
                    fid, sg_or_raw, intent = fut.result()

                    if raw:
                        if sg_or_raw:
                            all_raw[fid] = sg_or_raw
                            processed += 1
                    else:
                        if sg_or_raw and intent:
                            all_sg[fid] = sg_or_raw
                            all_intent[fid] = intent
                            processed += 1

                if raw:
                    with open(raw_path, "w") as f:
                        json.dump(all_raw, f)
                else:
                    with open(sg_path, "w") as f:
                        json.dump(all_sg, f)
                    with open(intent_path, "w") as f:
                        json.dump(all_intent, f)

                time.sleep(0.1)

        print(f"[INFO] GPU {gpu_id} finished. Processed {processed} frames.")

        if raw:
            return all_raw, None
        return all_sg, all_intent


if __name__ == "__main__":
    dataset_path = DEFAULT_DATASET_PATH
    model_path = DEFAULT_MODEL_PATH

    print(f"[INFO] Using dataset: {dataset_path}")
    print(f"[INFO] Using model:   {model_path}")
    print(f"[INFO] RAW_MODE_FROM_LAUNCHER = {RAW_MODE_FROM_LAUNCHER} "
          f"({'one-pass' if RAW_MODE_FROM_LAUNCHER else 'two-stage'})")
    print(f"[INFO] start={args.start}, end={args.end}, gpu_arg={args.gpu}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')}")

    infer = Qwen3LocalSGGInference(
        dataset_path=dataset_path,
        model_path=model_path,
        max_tokens=512,
        rate_limit=1,
        http_timeout=60,
        max_retries=2,
    )

    data = infer.load_data()
    keys = list(data.keys())
    sub_keys = keys[args.start: args.end]
    sub_data = {k: data[k] for k in sub_keys}

    infer.run_inference_dict(
        sub_data,
        overwrite=True,
        raw=RAW_MODE_FROM_LAUNCHER,
    )

    print("[INFO] Completed inference.")
