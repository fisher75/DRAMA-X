#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入必要的库：os用于文件操作，json处理数据，multiprocessing用于多进程
import os
import json
import multiprocessing as mp
from math import ceil
from subprocess import Popen, PIPE

# 引用同一个文件夹下的 worker 脚本中的配置，确保路径一致
from qwen3_local_sgg_intent import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    load_dataset_dict,
)

DATASET = DEFAULT_DATASET_PATH
NUM_GPUS = 2          # 【关键配置】这里决定用几张卡，需根据实际机器修改
RAW_MODE = True       # 【关键配置】模式开关 # True = 一阶段 one-pass (Risk+Action+Intent)
# RAW_MODE = False    # False = 两阶段 SceneGraph+Intent
# True (One-pass): 直接输入图，一次性输出 Risk + Action + Intent + Bbox
# False (Two-stage): 先生成 Scene Graph，再根据图生成 Intent（论文原始流程）

# 用统一的加载函数，兼容 JSON / JSONL
# 加载所有数据，算出总数，计算每个 GPU 分多少数据 (chunk size)
all_data_dict = load_dataset_dict(DATASET)
all_data = list(all_data_dict.items())

total = len(all_data)
chunk = ceil(total / NUM_GPUS) # 向上取整，保证最后一张卡也能分到数据


def run_worker(gpu_id, start, end):
    """
    工作进程函数：这将在主进程中被调用，用来启动子进程
    每个进程绑定一张 GPU，调用 qwen3_local_sgg_intent.py 跑子区间 [start, end)
    """
    # 【核心】这行代码强行指定该进程只看得到第 gpu_id 号卡，物理隔离显存
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 构造命令行指令，相当于在终端敲：python qwen3_local_sgg_intent.py --start 0 --end 100 ...
    cmd = [
        "python",
        "qwen3_local_sgg_intent.py",
        "--start", str(start),
        "--end", str(end),
        "--gpu", str(gpu_id), # 传参告诉脚本它是几号工
        "--raw_mode", str(int(RAW_MODE)), # 传参告诉脚本跑什么模式
    ]

    print(f"[GPU {gpu_id}] Running items {start} → {end}")

    # Popen 执行命令，PIPE 用于捕获子进程的打印输出
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate() # 等待子进程结束并获取输出
    # 打印子进程的日志，方便调试报错
    print(out.decode(), err.decode())


def merge_results():
    """
    所有 GPU 都跑完之后，把各自的 *_gpuX.json 合并成一个总文件。
    """
    # 定义输出目录，和 worker 脚本里的输出目录对应
    out_dir = os.path.join(os.path.dirname(DATASET), "outputs", "qwen3_local")

    # 根据模式生成文件名标签，确保合并时不搞混
    model_tag = os.path.basename(DEFAULT_MODEL_PATH.rstrip("/"))
    mode_tag = "onepass" if RAW_MODE else "twostage"
    tag = f"{model_tag}_{mode_tag}"

    if RAW_MODE:
        # One-pass 模式下，直接把所有 gpuX.json 合并成一个大的 JSON 文件
        merged = {}
        for i in range(NUM_GPUS):
            # 读取分片文件,合并每个 GPU 生成的文件
            gpu_file = os.path.join(out_dir, f"all_raw_{tag}_gpu{i}.json")
            if not os.path.exists(gpu_file):
                print(f"[WARN] file not found: {gpu_file}")
                continue
            with open(gpu_file, "r") as f:
                part = json.load(f)
                merged.update(part)
                print(f"[INFO] merged {len(part)} items from GPU {i}")

        # 写出合并后的总文件
        combined_path = os.path.join(out_dir, f"all_raw_{tag}.json")
        with open(combined_path, "w") as f:
            json.dump(merged, f)
        print(f"[INFO] combined raw results -> {combined_path}")
        print(f"[INFO] total merged items: {len(merged)}")

    else:
        # Two-stage 模式下，分开合并 Scene Graph 和 Intent 两个结果文件
        # (逻辑同上，只是合并两个字典)
        # ... (省略重复逻辑) ...
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
    # 多进程启动逻辑
    procs = []
    for i in range(NUM_GPUS):
        s = i * chunk
        e = min((i + 1) * chunk, total) # 计算当前 GPU 负责的数据范围 [s, e)
        # 创建进程，target 是 run_worker 函数
        p = mp.Process(target=run_worker, args=(i, s, e))
        p.start()
        procs.append(p)

    # join() 会阻塞主程序，直到所有 GPU 进程都跑完才继续往下走
    for p in procs:
        p.join()

    print("All GPUs completed.")
    # 最后一步：合并结果
    merge_results()
