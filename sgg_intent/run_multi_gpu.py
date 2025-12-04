#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import multiprocessing as mp
from math import ceil
from subprocess import Popen, PIPE

from qwen3_local_sgg_intent import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    load_dataset_dict,
)

DATASET = DEFAULT_DATASET_PATH
NUM_GPUS = 2          # 用几张卡
RAW_MODE = True       # True = 一阶段 one-pass (Risk+Action+Intent)
# RAW_MODE = False    # False = 两阶段 SceneGraph+Intent

# 用统一的加载函数，兼容 JSON / JSONL
all_data_dict = load_dataset_dict(DATASET)
all_data = list(all_data_dict.items())

total = len(all_data)
chunk = ceil(total / NUM_GPUS)


def run_worker(gpu_id, start, end):
    """
    每个进程绑定一张 GPU，调用 qwen3_local_sgg_intent.py 跑子区间 [start, end)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python",
        "qwen3_local_sgg_intent.py",
        "--start", str(start),
        "--end", str(end),
        "--gpu", str(gpu_id),
        "--raw_mode", str(int(RAW_MODE)),
    ]

    print(f"[GPU {gpu_id}] Running items {start} → {end}")

    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    print(out.decode(), err.decode())


def merge_results():
    """
    所有 GPU 都跑完之后，把各自的 *_gpuX.json 合并成一个总文件。
    """
    out_dir = os.path.join(os.path.dirname(DATASET), "outputs", "qwen3_local")

    model_tag = os.path.basename(DEFAULT_MODEL_PATH.rstrip("/"))
    mode_tag = "onepass" if RAW_MODE else "twostage"
    tag = f"{model_tag}_{mode_tag}"

    if RAW_MODE:
        merged = {}
        for i in range(NUM_GPUS):
            gpu_file = os.path.join(out_dir, f"all_raw_{tag}_gpu{i}.json")
            if not os.path.exists(gpu_file):
                print(f"[WARN] file not found: {gpu_file}")
                continue
            with open(gpu_file, "r") as f:
                part = json.load(f)
                merged.update(part)
                print(f"[INFO] merged {len(part)} items from GPU {i}")

        combined_path = os.path.join(out_dir, f"all_raw_{tag}.json")
        with open(combined_path, "w") as f:
            json.dump(merged, f)
        print(f"[INFO] combined raw results -> {combined_path}")
        print(f"[INFO] total merged items: {len(merged)}")

    else:
        merged_sg = {}
        merged_intent = {}

        for i in range(NUM_GPUS):
            sg_file = os.path.join(out_dir, f"all_scene_graphs_{tag}_gpu{i}.json")
            intent_file = os.path.join(out_dir, f"all_intent_jsons_{tag}_gpu{i}.json")

            if os.path.exists(sg_file):
                with open(sg_file, "r") as f:
                    part_sg = json.load(f)
                    merged_sg.update(part_sg)
                    print(f"[INFO] merged {len(part_sg)} scene graphs from GPU {i}")
            else:
                print(f"[WARN] missing scene graph file: {sg_file}")

            if os.path.exists(intent_file):
                with open(intent_file, "r") as f:
                    part_int = json.load(f)
                    merged_intent.update(part_int)
                    print(f"[INFO] merged {len(part_int)} intents from GPU {i}")
            else:
                print(f"[WARN] missing intent file: {intent_file}")

        sg_out = os.path.join(out_dir, f"all_scene_graphs_{tag}.json")
        intent_out = os.path.join(out_dir, f"all_intent_jsons_{tag}.json")

        with open(sg_out, "w") as f:
            json.dump(merged_sg, f)
        with open(intent_out, "w") as f:
            json.dump(merged_intent, f)

        print(f"[INFO] combined scene graphs -> {sg_out}")
        print(f"[INFO] combined intents -> {intent_out}")
        print(f"[INFO] total merged SG: {len(merged_sg)}, intents: {len(merged_intent)}")


if __name__ == "__main__":
    procs = []
    for i in range(NUM_GPUS):
        s = i * chunk
        e = min((i + 1) * chunk, total)
        p = mp.Process(target=run_worker, args=(i, s, e))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All GPUs completed.")
    merge_results()
