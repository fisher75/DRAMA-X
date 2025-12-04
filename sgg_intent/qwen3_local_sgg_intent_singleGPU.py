#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本脚本：本地 Qwen3-VL 版 SGG-Intent 推理
路径建议：/workspace/chz/code/DRAMA-X/sgg_intent/qwen3_local_sgg_intent.py

功能：
- 读取 DRAMA-X 的 updated_output.json（默认：../drama_intent/updated_output.json）
- 对每一帧图像：
  - （raw=False）两阶段：
      1) 生成 Scene Graph JSON
      2) 基于 Scene Graph 生成 Intent JSON
    结果保存到 outputs/qwen3_local/all_scene_graphs.json &
             outputs/qwen3_local/all_intent_jsons.json
  - （raw=True）一阶段：
      图像 → {"Risk", "Suggested_action", 每个对象的 Intent+Reason+BBox}
    结果保存到 outputs/qwen3_local/all_raw_op.json

使用前请确认：
- 已安装 transformers >= 4.43（支持 Qwen3-VL）
- 已下载 Qwen3-VL 模型到本地（修改 DEFAULT_MODEL_PATH）
"""

import os
import json
import time
import re
from io import BytesIO
from typing import Dict, Any, Tuple

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


# ====== 默认路径配置（请根据你的环境修改） ======
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

# 默认数据集 JSON 路径
DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "drama_intent", "updated_output.json")

# 默认模型路径（请改成你实际存放 Qwen3-VL 的本地目录）
# 例如："/workspace/models/VLM/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN3_VL_MODEL_PATH",
    "/workspace/models/VLM/Qwen3-VL-2B-Instruct",  # TODO: 请根据实际情况修改
)


class Qwen3LocalSGGInference:
    """
    本地 Qwen3-VL 版本的 SGG-Intent 推理器：
    - 不依赖 OpenRouter / OpenAI HTTP 接口
    - 完全在本地 GPU 上推理
    """

    def __init__(
        self,
        dataset_path: str,
        model_path: str,
        max_tokens: int = 512,
        rate_limit: int = 4,       # 本地推理默认不开太多并发，防止显存爆，2b选4,8b选2.
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
            trust_remote_code=True,   # 关键：使用模型自带代码
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,   # 关键：使用模型自带代码
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

    # ====== 基础函数 ======

    def load_data(self) -> Dict[str, Any]:
        with open(self.dataset_path, "r") as f:
            return json.load(f)

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
        """
        支持两种情况：
        - 本地路径：/xxx/xxx.jpg
        - HTTP(S) URL：以 http:// 或 https:// 开头
        """
        if image_path.startswith("http://") or image_path.startswith("https://"):
            resp = requests.get(image_path, timeout=self.http_timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
        return img

    def _generate_with_image_and_text(self, image: Image.Image, prompt: str) -> str:
        """
        使用 Qwen3-VL 进行一次带图像的生成：
        - 构造单轮对话：user: [<image>, <text prompt>]
        - 使用 processor.apply_chat_template 构造字符串
        - 再通过 processor(text=[...], images=[image]) 得到张量，送入 model.generate
        """
        # 构造 messages（注意：这里 image 部分只需要占位，真正的图像在后面传入 images 参数）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # 占位符，图像数据通过 processor(images=[...]) 提供
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 得到带特殊 token 的文本模板
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 将文本 + 图像 一起编码成模型输入
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

        # 只取生成部分（去掉 prompt token）
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
        """
        1) 从 raw 文本中提取第一个平衡的 JSON 对象
        2) 尝试 json.loads
        3) 失败则使用正则进行简单修复
        4) 若配置了 OPENAI_API_KEY + openai，可选地用 gpt-4o-mini 做最终 JSON 修复
        """
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
            # 若没有配平，直接返回从第一个 { 到结尾
            return s[start:]

        def _basic_fix(text: str) -> str:
            # 替换花括号中的奇怪引号
            text = text.replace("“", '"').replace("”", '"')
            text = text.replace("‘", "'").replace("’", "'")
            # 去掉 ```json fenced code
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            # Python 风格 True/False/None → JSON
            text = re.sub(r"\bTrue\b", "true", text)
            text = re.sub(r"\bFalse\b", "false", text)
            text = re.sub(r"\bNone\b", "null", text)
            # key 使用双引号：{'key': ...} → {"key": ...}
            text = re.sub(r"'(\w+)'\s*:", r'"\1":', text)
            # value 使用双引号：: 'xxx' → : "xxx"
            text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
            # 去掉 } 或 ] 前的尾逗号
            text = re.sub(r",\s*(?=[}\]])", "", text)
            # 尝试平衡大括号数量
            opens, closes = text.count("{"), text.count("}")
            if opens > closes:
                text += "}" * (opens - closes)
            elif closes > opens:
                text = "{" * (closes - opens) + text
            return text


        # 1) 如果有 <BEGIN_JSON>...</END_JSON>，优先取里面
        m = re.search(r"<BEGIN_JSON>(.*?)<END_JSON>", raw, re.S)
        fragment = m.group(1).strip() if m else raw.strip()

        # 2) 提取第一个平衡 JSON 块
        candidate = _extract_balanced(fragment)

        # 3) 直接尝试解析
        try:
            return pyjson.loads(candidate)
        except pyjson.JSONDecodeError:
            pass

        # 4) 简单修复后再试
        fixed = _basic_fix(candidate)
        try:
            return pyjson.loads(fixed)
        except pyjson.JSONDecodeError:
            pass

        # 5) 若可用则启用 gpt-4o-mini 做 JSON 修复，否则抛异常
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
            # 最后实在不行就抛异常，交给上层处理
            raise pyjson.JSONDecodeError(
                "Failed to parse JSON after basic fixes and no JSON fallback available.",
                fixed,
                0,
            )

    # ====== 三个核心任务函数 ======

    def process_image_for_scene_graph(self, image_path: str) -> Dict[str, Any]:
        """
        场景图生成：图像 + scene_graph_prompt → Scene Graph JSON
        """
        try:
            img = self._load_image(image_path)
            prompt = (
                f"{self.scene_graph_prompt_template}\n\n"
                "Analyse this image and output ONLY one JSON object."
            )
            out_text = self._generate_with_image_and_text(img, prompt)

            # 去掉 ```json 包裹
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
        """
        Intent 生成：图像 + scene_graph + intent_prompt → Intent JSON
        """
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
        """
        一阶段版本：图像 + all_gen_prompt → {Risk, Suggested_action, 对象Intent...} JSON
        """
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
        """
        处理单帧：
        - raw=False: 生成 (scene_graph, intent)
        - raw=True : 生成 (all_in_one_result, None)
        """
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

    def run_inference(self, limit: int = None, overwrite: bool = True, raw: bool = False):
        """
        主入口：对数据集进行批量推理
        - limit: 只处理前 N 个样本（调试用）
        - overwrite: 是否覆盖已有结果（False 时会做断点续跑）
        - raw: True → 一阶段（Risk+Action+Intent）；False → 两阶段（SceneGraph+Intent）
        """
        data = self.load_data()
        out_dir = os.path.join(os.path.dirname(self.dataset_path), "outputs", "qwen3_local")
        os.makedirs(out_dir, exist_ok=True)

        items = list(data.items())
        print(f"[INFO] Original num items: {len(items)}")
        if limit is not None:
            items = items[:limit]

        if raw:
            raw_path = os.path.join(out_dir, "all_raw_op.json")
            if not overwrite and os.path.exists(raw_path):
                with open(raw_path, "r") as f:
                    all_raw = json.load(f)
            else:
                all_raw = {}

            if not overwrite:
                items = [(fid, fdata) for fid, fdata in items if fid not in all_raw]
        else:
            sg_path = os.path.join(out_dir, "all_scene_graphs.json")
            intent_path = os.path.join(out_dir, "all_intent_jsons.json")

            if not overwrite and os.path.exists(sg_path):
                with open(sg_path, "r") as f:
                    all_sg = json.load(f)
            else:
                all_sg = {}
            if not overwrite and os.path.exists(intent_path):
                with open(intent_path, "r") as f:
                    all_intent = json.load(f)
            else:
                all_intent = {}

            if not overwrite:
                items = [
                    (fid, fdata)
                    for fid, fdata in items
                    if fid not in all_sg or fid not in all_intent
                ]

        print(f"[INFO] Remaining num items: {len(items)}")

        # 本地模型并发不宜太大，concur = max(1, rate_limit)
        concur = max(1, self.rate_limit)
        chunks = [items[i: i + concur] for i in range(0, len(items), concur)]

        processed = 0
        with ThreadPoolExecutor(max_workers=concur) as exe:
            for batch in tqdm(chunks, desc="Processing batches"):
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

                # 每个 batch 写一次文件，防止中途崩溃丢进度
                if raw:
                    with open(raw_path, "w") as f:
                        json.dump(all_raw, f)
                else:
                    with open(sg_path, "w") as f:
                        json.dump(all_sg, f)
                    with open(intent_path, "w") as f:
                        json.dump(all_intent, f)

                # 简单节流（如果你觉得不需要可以注释掉）
                time.sleep(0.2)

        print(f"[INFO] Done! Processed {processed} frames.")
        if raw:
            return all_raw, None
        return all_sg, all_intent


if __name__ == "__main__":
    # ========= 入口示例 =========
    dataset_path = DEFAULT_DATASET_PATH
    model_path = DEFAULT_MODEL_PATH

    print(f"[INFO] Using dataset: {dataset_path}")
    print(f"[INFO] Using model:   {model_path}")

    infer = Qwen3LocalSGGInference(
        dataset_path=dataset_path,
        model_path=model_path,
        max_tokens=512,
        rate_limit=1,      # 建议先单线程跑稳定，再考虑开并发
        http_timeout=60,
        max_retries=2,
    )

    # 调试建议：先 limit=4，确认 JSON 结构正常
    # raw=True: 一阶段 (Risk + Suggested_action + Intent)
    # raw=False: 两阶段 (SceneGraph + Intent)
    all_raw, _ = infer.run_inference(
        limit=None, # 测试用的limit=4，现在修改为None变成全量。
        overwrite=True,
        raw=True,
    )

    print("[INFO] Completed inference (raw=True, limit=None).")