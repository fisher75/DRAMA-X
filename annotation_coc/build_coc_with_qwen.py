#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_coc_with_qwen.py

功能：
- 读取 build_structured_facts.py 生成的 drama_x_structured.jsonl
- 使用本地 Qwen3-VL 为每条样本生成 CoC 风格 reasoning：
    {
      "driving_decision": "GO|SLOW|STOP",
      "critical_components": {...},
      "coc_trace": "Because ... therefore ..."
    }
- 将生成结果与原 structured facts 合并，写入新的 jsonl

模式：
1) 单进程（默认）：device_map="auto"（模型并行）
   - 用 CUDA_VISIBLE_DEVICES 控制可见 GPU 数量

2) 数据并行（--data_parallel）：多进程，每张卡一份模型，处理切片
   - 对 2B + 大批量推理通常更快、更稳

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python build_coc_with_qwen.py \
  --structured_path ... \
  --output_path ... \
  --model_path /workspace/models/VLM/Qwen3-VL-2B-Instruct \
  --data_parallel \
  --preserve_order

实际使用：
CUDA_VISIBLE_DEVICES=0,1,2
python build_coc_with_qwen.py --structured_path /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_structured.jsonl --output_path /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b.jsonl --model_path /workspace/models/VLM/Qwen3-VL-2B-Instruct --data_parallel --preserve_order

关键修复：
- 推理路径严格复刻你能跑通的 qwen3_local_sgg_intent_singleGPU.py
- 默认不手动修改 image_grid_thw
- 仅当 processor 输出的 image_grid_thw “明显坏掉”(None/空/含0) 时才兜底修复
- 增强 debug：出错自动打印关键输入张量信息
"""

import os
import re
import json
import math
import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import requests
from PIL import Image

import torch
import torch.multiprocessing as mp
from transformers import AutoProcessor, AutoModelForVision2Seq


# -----------------------------
# JSON 提取（平衡括号，更稳）
# -----------------------------
def extract_first_balanced_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    t = text.strip()
    if t.startswith("```json"):
        t = t[7:].strip()
    if t.endswith("```"):
        t = t[:-3].strip()

    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    end = -1
    for i, ch in enumerate(t[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None

    snippet = t[start:end + 1]

    # 轻量修复
    snippet = snippet.replace("“", '"').replace("”", '"')
    snippet = snippet.replace("‘", "'").replace("’", "'")
    snippet = re.sub(r"\bTrue\b", "true", snippet)
    snippet = re.sub(r"\bFalse\b", "false", snippet)
    snippet = re.sub(r"\bNone\b", "null", snippet)
    snippet = re.sub(r",\s*(?=[}\]])", "", snippet)

    try:
        return json.loads(snippet)
    except Exception:
        return None


# -----------------------------
# dtype 选择：优先 bf16，不行就 fp16
# -----------------------------
def pick_torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# -----------------------------
# 图像加载（HTTP / 本地）
# -----------------------------
def load_image(image_path: str, timeout: int = 60) -> Image.Image:
    if not image_path:
        raise ValueError("Empty image_path")
    if image_path.startswith("http://") or image_path.startswith("https://"):
        resp = requests.get(image_path, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_path).convert("RGB")


# -----------------------------
# 仅在 processor 输出“坏 grid”时兜底修复
# （默认不改，避免破坏 Qwen3-VL 内部一致性）
# -----------------------------
def _get_patch_size_from_model(model) -> int:
    vc = getattr(model.config, "vision_config", None)
    ps = getattr(vc, "patch_size", None) if vc is not None else None
    if isinstance(ps, (list, tuple)) and len(ps) > 0:
        ps = ps[0]
    try:
        return int(ps) if ps is not None else 14
    except Exception:
        return 14


def _is_bad_grid(g) -> bool:
    if g is None:
        return True
    if not torch.is_tensor(g):
        return True
    if g.numel() == 0:
        return True
    if g.ndim != 2 or g.shape[-1] != 3:
        return True
    if (g <= 0).any().item():
        return True
    return False


def fix_bad_image_grid_thw_if_needed(inputs, model) -> bool:
    """
    返回：是否做了修复
    只在 image_grid_thw 明显坏掉时修复。
    """
    grid = inputs.get("image_grid_thw", None)
    pv = inputs.get("pixel_values", None)

    if pv is None:
        return False

    if not _is_bad_grid(grid):
        return False

    # 兜底：按 pixel_values 的 H/W 推断 gh/gw
    # 常见 pv: [B, C, H, W]
    if torch.is_tensor(pv) and pv.ndim == 4:
        patch = _get_patch_size_from_model(model)
        H, W = int(pv.shape[-2]), int(pv.shape[-1])
        gh, gw = max(H // patch, 1), max(W // patch, 1)
        fixed = torch.tensor([[1, gh, gw]], dtype=torch.long)
        inputs["image_grid_thw"] = fixed
        return True

    # 兜底再兜底
    inputs["image_grid_thw"] = torch.tensor([[1, 1, 1]], dtype=torch.long)
    return True

def ensure_image_grid_thw_consistent(inputs, image: Image.Image) -> bool:
    """
    强制保证 image_grid_thw 和 pixel_values 的 token 数一致：
      prod(t,h,w) == N (N = pixel_values.shape[1])
    若不一致，就根据 N 的因子 + 原图宽高比，选一个最合理的 (h,w)
    返回：是否发生了修改
    """
    pv = inputs.get("pixel_values", None)
    if pv is None or (not torch.is_tensor(pv)):
        return False

    # 只处理 (B,N,C) 或 (N,C)
    if pv.ndim == 2:
        N = int(pv.shape[0])
    elif pv.ndim == 3:
        N = int(pv.shape[1])
    else:
        return False

    grid = inputs.get("image_grid_thw", None)

    def prod_ok(g) -> bool:
        if g is None or (not torch.is_tensor(g)) or g.numel() != 3:
            return False
        g = g.view(-1)
        if (g <= 0).any().item():
            return False
        return int(g[0] * g[1] * g[2]) == N

    # 如果 grid 已经正确，直接返回
    if prod_ok(grid):
        return False

    # 否则重算：t=1，找 h*w=N 的因子对
    aspect = (image.width / max(image.height, 1)) if image is not None else 1.0

    # 找所有因子对
    pairs = []
    for h in range(1, int(math.sqrt(N)) + 1):
        if N % h == 0:
            w = N // h
            pairs.append((h, w))

    if not pairs:
        # 极端兜底
        inputs["image_grid_thw"] = torch.tensor([[1, 1, N]], dtype=torch.long)
        return True

    # 选最符合原图宽高比的 (h,w)
    best_h, best_w = min(pairs, key=lambda hw: abs((hw[1] / hw[0]) - aspect))

    inputs["image_grid_thw"] = torch.tensor([[1, best_h, best_w]], dtype=torch.long)
    return True



def debug_dump_inputs(inputs, prefix: str = "[DEBUG]"):
    keys = list(inputs.keys())
    print(f"{prefix} inputs keys = {keys}")
    for k in keys:
        v = inputs[k]
        if torch.is_tensor(v):
            info = f"shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
            if v.numel() > 0 and v.dtype in (torch.float16, torch.float32, torch.bfloat16):
                info += f" min={v.min().item():.4f} max={v.max().item():.4f}"
            if v.numel() > 0 and v.dtype in (torch.int32, torch.int64, torch.long):
                info += f" min={v.min().item()} max={v.max().item()}"
            print(f"{prefix} {k}: {info}")
        else:
            print(f"{prefix} {k}: type={type(v)}")


# -----------------------------
# CoC 生成器（推理路径对齐你的可用脚本）
# -----------------------------
class Qwen3VLCoCGenerator:
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 256,
        http_timeout: int = 60,
        device: Optional[torch.device] = None,
        device_map: Optional[str] = "auto",
        enable_grid_fix: bool = False,
        verbose_debug: bool = False,
    ) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.http_timeout = http_timeout
        self.enable_grid_fix = enable_grid_fix
        self.verbose_debug = verbose_debug

        dtype = pick_torch_dtype()

        print(f"[INFO] Loading Qwen3-VL processor from: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        print(f"[INFO] Loading Qwen3-VL model from: {self.model_path}")
        if device_map is None:
            # 数据并行：每进程单卡
            assert device is not None, "device_map=None 时必须传 device"
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            self.device = device
            self.model.to(self.device)
        else:
            # 单进程模型并行
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device_map,
            )
            self.device = self.model.device

        self.model.eval()
        print(f"[INFO] Model loaded. main device = {self.device}")

    def build_user_prompt(self, sample: Dict[str, Any]) -> str:
        sid = sample.get("sample_id", sample.get("id", ""))
        risk_label = sample.get("risk_label", sample.get("Risk", "low"))
        suggested_action = sample.get("suggested_action_raw", sample.get("Suggested_action", ""))

        vru_lines = []
        for v in sample.get("vru_list", []):
            vid = v.get("vru_id", "")
            vtype = v.get("type", "")
            intents = ", ".join(v.get("intent_list", [])) or "unknown intent"
            pos_cat = v.get("position_category", "other")
            dist = v.get("distance_level", "unknown")
            desc = v.get("description", "")
            vru_lines.append(
                f"- {vid}: type={vtype}, intent={intents}, position_category={pos_cat}, distance={dist}, desc={desc}"
            )
        vru_block = "\n".join(vru_lines) if vru_lines else "No VRUs detected."

        prompt = f"""
You are an autonomous driving safety expert.
You are given one traffic image sample from a VRU risk dataset (DRAMA-X) and some structured annotations.

Sample ID: {sid}
Overall risk label (from dataset): {risk_label}
Suggested action (from dataset): {suggested_action}

List of vulnerable road users (VRUs) with attributes:
{vru_block}

Your task:
1) Decide one high-level driving decision for the ego vehicle, choose strictly from:
- "GO"
- "SLOW"
- "STOP"

2) Summarize the critical components that influence this decision, in a structured JSON object.
This should include at least the primary VRU (id, type, intent, distance, position) and optionally other relevant context.

3) Write a short Chain-of-Causation reasoning (1-3 sentences) that explains the decision.
Use a clear "Because ... therefore ..." style.

Important constraints:
- Only use information that could be inferred from the image and the annotations above.
- Do NOT invent objects or events that are not supported.

Output format:
Return a single JSON object with exactly the following keys:
- "driving_decision": one of "GO", "SLOW", "STOP".
- "critical_components": a JSON object describing key VRUs and scene factors.
- "coc_trace": a short textual explanation (1-3 sentences) in "Because ... therefore ..." pattern.
Do not add any extra keys besides these three.
""".strip()

        return prompt

    @torch.no_grad()
    def generate_coc_for_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1) load image
        image_path = sample.get("image_path", "")
        image = load_image(image_path, timeout=self.http_timeout)

        # 2) build prompt & chat template
        user_prompt = self.build_user_prompt(sample)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 3) processor 编码（先在 CPU）
        inputs = self.processor(text=[chat_text], images=[image], return_tensors="pt")

        # ---- (A) 先在 CPU 上把视觉输入形状修好（最关键）----
        # Qwen3-VL 某些版本 processor 会给 pixel_values=(N,C)，必须补 batch -> (1,N,C)
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            if inputs["pixel_values"].dim() == 2:
                inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)

        # image_grid_thw 也确保是 (1,3)
        if "image_grid_thw" in inputs and isinstance(inputs["image_grid_thw"], torch.Tensor):
            if inputs["image_grid_thw"].dim() == 1:
                inputs["image_grid_thw"] = inputs["image_grid_thw"].unsqueeze(0)

        # ---- (B) 强制保证 grid 与 token 数一致（最关键，建议永远开启）----
        fixed = ensure_image_grid_thw_consistent(inputs, image=image)
        if fixed and self.verbose_debug:
            print("[FIX] image_grid_thw adjusted to match pixel_values tokens")

        # （可选）如果你还想保留旧的坏grid兜底逻辑，也可以再跑一次
        # if self.enable_grid_fix:
        #     fix_bad_image_grid_thw_if_needed(inputs, model=self.model)


        # ---- (C) 再逐 key 搬到主 device（别用 BatchEncoding.to 整体搬）----
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device, non_blocking=True)

        # === hard check: print grid/N/product ===
        pv = inputs["pixel_values"]
        N = int(pv.shape[1] if pv.ndim == 3 else pv.shape[0])

        g = inputs.get("image_grid_thw", None)
        print("[CHECK] pixel_values N =", N)
        if torch.is_tensor(g):
            gv = g.view(-1).detach().cpu().long().tolist()
            print("[CHECK] image_grid_thw =", gv, " product =", int(gv[0]*gv[1]*gv[2]))
            assert gv[0] > 0 and gv[1] > 0 and gv[2] > 0, "grid has non-positive entries!"
            assert int(gv[0]*gv[1]*gv[2]) == N, f"grid product != N: {gv} vs N={N}"
        else:
            print("[CHECK] image_grid_thw is NOT tensor:", type(g))
            raise RuntimeError("image_grid_thw missing / not tensor")


        # =========================
        # ✅ 修复 3：forward 自检 —— 就加在这里
        # =========================
        if self.verbose_debug and "image_grid_thw" in inputs:
            print("[DEBUG] image_grid_thw =", inputs["image_grid_thw"].cpu().tolist())
        if self.verbose_debug and "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            N = pv.shape[1] if pv.ndim == 3 else pv.shape[0]
            print("[DEBUG] pixel_values tokens N =", int(N))
        
        try:
            _ = self.model(**inputs, return_dict=True)
        except Exception as e:
            print(f"[SELF-CHECK] forward failed on sample_id={sample.get('sample_id')}")
            # 你也可以在这里 dump shapes，定位更快
            for kk, vv in inputs.items():
                if isinstance(vv, torch.Tensor):
                    print(f"[DUMP] {kk}: shape={tuple(vv.shape)} dtype={vv.dtype} device={vv.device}")
            raise e  # 让它直接报栈，方便你贴给我

        # 7) forward 过了，才进入 generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

        # 8) decode & parse
        input_ids = inputs["input_ids"]
        gen_ids = generated_ids[:, input_ids.shape[1]:]
        out_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        coc_json = extract_first_balanced_json(out_text)
        return coc_json



# -----------------------------
# 数据并行 worker
# -----------------------------
def dp_worker(rank: int, world_size: int, args_dict: Dict[str, Any]) -> None:
    structured_path = args_dict["structured_path"]
    output_path = args_dict["output_path"]
    model_path = args_dict["model_path"]
    max_samples = args_dict["max_samples"]
    max_tokens = args_dict["max_tokens"]
    http_timeout = args_dict["http_timeout"]
    preserve_order = args_dict["preserve_order"]
    enable_grid_fix = args_dict["enable_grid_fix"]
    verbose_debug = args_dict["verbose_debug"]

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    gen = Qwen3VLCoCGenerator(
        model_path=model_path,
        max_tokens=max_tokens,
        http_timeout=http_timeout,
        device=device,
        device_map=None,  # 每进程单卡
        enable_grid_fix=enable_grid_fix,
        verbose_debug=verbose_debug,
    )

    tmp_path = f"{output_path}.rank{rank}.tmp"
    num_in, num_ok = 0, 0

    with open(structured_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if max_samples > 0 and idx >= max_samples:
                break
            if idx % world_size != rank:
                continue

            line = line.strip()
            if not line:
                continue

            num_in += 1
            sample = json.loads(line)
            coc = gen.generate_coc_for_sample(sample)
            if coc is None:
                continue

            merged = {**sample, **coc}
            if preserve_order:
                merged["_line_index"] = idx

            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            num_ok += 1

            if num_in % 10 == 0:
                print(f"[RANK {rank}] processed={num_in}, ok={num_ok}")

    print(f"[RANK {rank}] DONE. tmp={tmp_path} in={num_in} ok={num_ok}")


def merge_tmp_files(output_path: str, world_size: int, preserve_order: bool) -> None:
    tmp_files = [f"{output_path}.rank{r}.tmp" for r in range(world_size)]
    records: List[Dict[str, Any]] = []

    for fp in tmp_files:
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

    if preserve_order:
        records.sort(key=lambda x: x.get("_line_index", 10**18))
        for r in records:
            r.pop("_line_index", None)

    with open(output_path, "w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    for fp in tmp_files:
        try:
            if os.path.exists(fp):
                os.remove(fp)
        except Exception:
            pass

    print(f"[DONE] merged {len(records)} records -> {output_path}")


# -----------------------------
# 主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Build CoC-style reasoning for DRAMA-X using Qwen3-VL")
    parser.add_argument("--structured_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--http_timeout", type=int, default=60)

    parser.add_argument("--data_parallel", action="store_true",
                        help="Enable data-parallel multi-process inference (one process per visible GPU).")
    parser.add_argument("--preserve_order", action="store_true",
                        help="Preserve input order by sorting with _line_index when merging DP outputs.")

    # ✅ 默认关闭：只有你确认确实存在坏 grid 样本再开
    parser.add_argument("--enable_grid_fix", action="store_true",
                        help="Only fix image_grid_thw when processor output is clearly bad (None/0/empty).")

    # ✅ debug：打印 inputs 的关键张量信息
    parser.add_argument("--verbose_debug", action="store_true",
                        help="Dump model inputs (shapes/dtypes) before generation for debugging.")

    args = parser.parse_args()

    structured_path = str(Path(args.structured_path).expanduser())
    output_path = str(Path(args.output_path).expanduser())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if args.data_parallel:
        world_size = torch.cuda.device_count()
        if world_size <= 0:
            raise RuntimeError("No CUDA device visible. Turn off --data_parallel or set CUDA_VISIBLE_DEVICES properly.")

        print(f"[INFO] Data-parallel enabled. world_size={world_size}")
        args_dict = {
            "structured_path": structured_path,
            "output_path": output_path,
            "model_path": args.model_path,
            "max_samples": args.max_samples,
            "max_tokens": args.max_tokens,
            "http_timeout": args.http_timeout,
            "preserve_order": args.preserve_order,
            "enable_grid_fix": args.enable_grid_fix,
            "verbose_debug": args.verbose_debug,
        }

        mp.spawn(
            dp_worker,
            args=(world_size, args_dict),
            nprocs=world_size,
            join=True,
        )

        merge_tmp_files(output_path, world_size, preserve_order=args.preserve_order)
        return

    # 默认：单进程（模型并行）
    print("[INFO] Single-process (model-parallel if multiple GPUs visible) mode")
    gen = Qwen3VLCoCGenerator(
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        http_timeout=args.http_timeout,
        device=None,
        device_map="auto",
        enable_grid_fix=args.enable_grid_fix,
        verbose_debug=args.verbose_debug,
    )

    num_in, num_ok = 0, 0
    with open(structured_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if args.max_samples > 0 and idx >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue

            num_in += 1
            sample = json.loads(line)
            coc = gen.generate_coc_for_sample(sample)
            if coc is None:
                continue

            merged = {**sample, **coc}
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            num_ok += 1

            if num_in % 10 == 0:
                print(f"[INFO] processed={num_in}, ok={num_ok}")

    print(f"[DONE] total_in={num_in}, ok={num_ok}")
    print(f"[DONE] saved to: {output_path}")


if __name__ == "__main__":
    main()
