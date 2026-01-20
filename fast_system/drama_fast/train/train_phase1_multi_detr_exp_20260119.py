"""Phase-1 training: Video Swin-T backbone + (single/multi)-query joint head.

This script reuses the existing interfaces:
- Dataset: drama_fast.dataset_phase1.DramaFastDataset (+default_collate)
- Model  : drama_fast.models.model_phase1.FastSystemPhase1

Key upgrades vs v2:
- Optional multi-query training (num_queries>1) to reduce "wrong-object" boxes.
- DETR-style box loss: SmoothL1 + (1-GIoU).
- Greedy matching between queries and top-k GT targets (fast, avoids scipy dependency).
- Proper no-object handling for unmatched queries (risk -> 0).

Single GPU:
  python -m drama_fast.train.train_phase1 --train_jsonl ... --val_jsonl ... --data_root ... --out_dir ...

20260119:
你现在的标注是 drama_x_fast_sup_v3_topk5.jsonl，所以建议：
--topk_targets 5
--num_queries 6 或 8（略大于 K，给模型多一点自由度；你也可以先用 5）

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DRAMA_DATA_ROOT=/data2/automan/data/drama_data

python -m drama_fast.train.train_phase1 \
  --train_jsonl ./splits_v3/train.jsonl \
  --val_jsonl   ./splits_v3/val.jsonl \
  --data_root   $DRAMA_DATA_ROOT \
  --out_dir     ./runs/phase1_swin_t_ddp_v3_topk5_q8 \
  --num_frames 8 --img_size 224 --stride 2 \
  --topk_targets 5 \
  --num_queries 8 \
  --batch_size 16 \
  --epochs 50 \
  --lr 2e-4 --weight_decay 0.05 \
  --amp \
  --pretrained \
  --ddp --gpus 4 \
  --wandb --wandb_run_name "SwinT-v3-topk5-q8"

20260118:
# 1. 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DRAMA_DATA_ROOT=/data2/automan/data/drama_data

# 2. 启动命令
# 注意：
# --batch_size 8: 每张卡 8，总 Batch Size = 32 (8*4)。显存如果爆了改成 4。
# --wandb: 开启 WandB 记录
# --pretrained: 必须加，提升收敛速度
python -m drama_fast.train.train_phase1 \
  --train_jsonl ./splits_v1/train.jsonl \
  --val_jsonl   ./splits_v1/val.jsonl \
  --data_root   $DRAMA_DATA_ROOT \
  --out_dir     ./runs/phase1_swin_t_ddp_full \
  --num_frames 8 --img_size 224 --stride 2 \
  --batch_size 16 \
  --epochs 50 \
  --lr 2e-4 --weight_decay 0.05 \
  --amp \
  --pretrained \
  --ddp --gpus 4 \
  --wandb \
  --wandb_run_name "SwinT-Full-Run-v2"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from drama_fast.dataset_phase1 import DramaFastDataset, default_collate
from drama_fast.models.model_phase1 import FastSystemPhase1


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def maybe_init_wandb(args: argparse.Namespace, out_dir: Path):
    """Init Weights & Biases on main process only. Safe for DDP(spawn)."""
    if not getattr(args, "wandb", False):
        return None

    # Disable W&B in non-main processes to avoid duplicate runs.
    if not is_main():
        os.environ.setdefault("WANDB_MODE", "disabled")
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[WARN] W&B enabled but import failed: {e}")
        return None

    run_name = getattr(args, "wandb_run_name", None)
    if not run_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{out_dir.name}-{ts}"

    tags = []
    tag_str = getattr(args, "wandb_tags", "")
    if tag_str:
        tags = [t.strip() for t in tag_str.split(",") if t.strip()]

    wandb.init(
        entity=getattr(args, "wandb_entity", None),
        project=getattr(args, "wandb_project", "aumovio"),
        name=run_name,
        dir=str(out_dir),
        config=vars(args),
        tags=tags,
    )
    return wandb


def ddp_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


# -----------------------------
# Box ops
# -----------------------------

def _box_area_xyxy(box: torch.Tensor) -> torch.Tensor:
    # box: [...,4] xyxy
    w = (box[..., 2] - box[..., 0]).clamp(min=0)
    h = (box[..., 3] - box[..., 1]).clamp(min=0)
    return w * h


def bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU for normalized xyxy boxes.

    a, b: [B,4] in xyxy, each coord in [0,1]
    returns: [B]
    """
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    ix1 = torch.max(ax1, bx1)
    iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2)
    iy2 = torch.min(ay2, by2)

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)

    union = area_a + area_b - inter
    return inter / (union + eps)


def bbox_giou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Generalized IoU for normalized xyxy boxes.

    a, b: [B,4]
    returns: [B]
    """
    iou = bbox_iou_xyxy(a, b, eps=eps)

    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    cx1 = torch.min(ax1, bx1)
    cy1 = torch.min(ay1, by1)
    cx2 = torch.max(ax2, bx2)
    cy2 = torch.max(ay2, by2)

    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)

    # union (reuse from IoU computation)
    ix1 = torch.max(ax1, bx1)
    iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2)
    iy2 = torch.min(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a + area_b - inter

    giou = iou - (c_area - union) / (c_area + eps)
    return giou


def pairwise_giou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise GIoU.

    boxes1: [Q,4], boxes2: [K,4]
    returns: [Q,K]
    """
    a = boxes1[:, None, :]  # [Q,1,4]
    b = boxes2[None, :, :]  # [1,K,4]

    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    ix1 = torch.max(ax1, bx1)
    iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2)
    iy2 = torch.min(ay2, by2)

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a + area_b - inter

    iou = inter / (union + eps)

    cx1 = torch.min(ax1, bx1)
    cy1 = torch.min(ay1, by1)
    cx2 = torch.max(ax2, bx2)
    cy2 = torch.max(ay2, by2)

    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)

    giou = iou - (c_area - union) / (c_area + eps)
    return giou


def greedy_match(cost: torch.Tensor) -> Tuple[list[int], list[int]]:
    """Greedy bipartite matching on a small cost matrix.

    cost: [Q, K]
    returns: matched_q, matched_k (same length)

    NOTE: This is not optimal Hungarian, but works well when Q and K are small (<=5)
    and is much cheaper / dependency-free.
    """
    Q, K = cost.shape
    remaining_q = set(range(Q))
    remaining_k = set(range(K))
    mq: list[int] = []
    mk: list[int] = []

    # Convert to CPU for fast Python loops when Q,K very small.
    c = cost.detach().cpu()

    while remaining_q and remaining_k:
        best = None
        best_val = float("inf")
        for q in remaining_q:
            row = c[q]
            for k in remaining_k:
                v = float(row[k].item())
                if v < best_val:
                    best_val = v
                    best = (q, k)
        if best is None:
            break
        q, k = best
        mq.append(q)
        mk.append(k)
        remaining_q.remove(q)
        remaining_k.remove(k)

    return mq, mk


def compute_loss_multiquery(
    pred_boxes: torch.Tensor,
    pred_risks: torch.Tensor,
    gt_boxes_topk: torch.Tensor,
    gt_risks_topk: torch.Tensor,
    gt_mask_topk: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (total, box, risk) losses.

    pred_boxes:    [B,Q,4]
    pred_risks:    [B,Q]
    gt_boxes_topk: [B,K,4]
    gt_risks_topk: [B,K]
    gt_mask_topk:  [B,K] (bool)
    """
    # 1. 为了数值稳定性和避免 AMP 报错，建议把所有用于 Loss 计算的 Tensor 转为 float32
    # 这比只改一行 assignment 更彻底，能防止 L1/GIoU 在 float16 下溢出
    pred_boxes = pred_boxes.float()
    pred_risks = pred_risks.float()
    
    # 修正变量名，对应函数参数
    gt_boxes_topk = gt_boxes_topk.float()
    gt_risks_topk = gt_risks_topk.float()
    gt_mask_topk = gt_mask_topk.float()
    
    B, Q, _ = pred_boxes.shape
    K = gt_boxes_topk.shape[1]

    total_box = pred_boxes.new_tensor(0.0)
    total_risk = pred_boxes.new_tensor(0.0)
    denom = 0

    for b in range(B):
        pb = pred_boxes[b]         # [Q,4]
        pr = pred_risks[b]         # [Q]
        gb_all = gt_boxes_topk[b]  # [K,4]
        gr_all = gt_risks_topk[b]  # [K]
        mask = gt_mask_topk[b]     # [K]

        valid_idx = torch.nonzero(mask, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            # No GT objects: only train risks to be 0 for all queries
            risk_t = torch.zeros((Q,), device=pr.device, dtype=pr.dtype)
            w = torch.full((Q,), float(args.risk_noobj_weight), device=pr.device, dtype=pr.dtype)
            risk_loss = (F.smooth_l1_loss(pr, risk_t, reduction="none") * w).mean()
            total_risk = total_risk + risk_loss
            denom += 1
            continue

        gb = gb_all[valid_idx]  # [Kv,4]
        gr = gr_all[valid_idx]  # [Kv]
        Kv = gb.shape[0]

        # --- cost matrix ---
        l1 = torch.abs(pb[:, None, :] - gb[None, :, :]).sum(-1)  # [Q,Kv]
        giou = pairwise_giou_xyxy(pb, gb)                        # [Q,Kv]
        giou_loss = 1.0 - giou
        risk_cost = torch.abs(pr[:, None] - gr[None, :])         # [Q,Kv]

        cost = args.box_l1_weight * l1 + args.box_giou_weight * giou_loss + args.match_risk_weight * risk_cost

        mq, mk_local = greedy_match(cost)
        if len(mq) == 0:
            # fallback: no matches, treat as no-obj
            risk_t = torch.zeros((Q,), device=pr.device, dtype=pr.dtype)
            w = torch.full((Q,), float(args.risk_noobj_weight), device=pr.device, dtype=pr.dtype)
            risk_loss = (F.smooth_l1_loss(pr, risk_t, reduction="none") * w).mean()
            total_risk = total_risk + risk_loss
            denom += 1
            continue

        mq_t = torch.tensor(mq, device=pb.device, dtype=torch.long)
        mk_t = torch.tensor(mk_local, device=pb.device, dtype=torch.long)

        pb_m = pb[mq_t]  # [M,4]
        gb_m = gb[mk_t]  # [M,4]

        # Box losses (matched)
        box_l1 = torch.abs(pb_m - gb_m).sum(-1).mean()
        box_giou = (1.0 - bbox_giou_xyxy(pb_m, gb_m)).mean()
        box_loss = args.box_l1_weight * box_l1 + args.box_giou_weight * box_giou

        # Risk loss for all queries, with no-obj downweight
        risk_t = torch.zeros((Q,), device=pr.device, dtype=pr.dtype)
        risk_t[mq_t] = gr[mk_t]
        w = torch.full((Q,), float(args.risk_noobj_weight), device=pr.device, dtype=pr.dtype)
        w[mq_t] = 1.0
        risk_loss = (F.smooth_l1_loss(pr, risk_t, reduction="none") * w).mean()

        total_box = total_box + box_loss
        total_risk = total_risk + risk_loss
        denom += 1

    denom = max(1, denom)
    total_box = total_box / denom
    total_risk = total_risk / denom

    total = args.lambda_box * total_box + args.lambda_risk * total_risk
    return total, total_box, total_risk


@dataclass
class TrainState:
    epoch: int
    best_val_loss: float
    global_step: int


def save_checkpoint(
    out_dir: Path,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    args: argparse.Namespace,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # unwrap DDP
    m = model.module if isinstance(model, DDP) else model

    ckpt = {
        "model": m.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": asdict(state),
        "args": vars(args),
    }
    if scaler is not None:
        try:
            ckpt["scaler"] = scaler.state_dict()
        except Exception:
            pass

    torch.save(ckpt, out_dir / name)


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> TrainState:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)

    # handle DDP saved keys
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    m = model.module if isinstance(model, DDP) else model
    m.load_state_dict(sd, strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    st = ckpt.get("state", {"epoch": 0, "best_val_loss": float("inf"), "global_step": 0})
    return TrainState(
        epoch=int(st.get("epoch", 0)),
        best_val_loss=float(st.get("best_val_loss", float("inf"))),
        global_step=int(st.get("global_step", 0)),
    )


@torch.no_grad()
def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_box = 0.0
    total_risk = 0.0

    # primary metrics
    total_iou_primary = 0.0
    total_iou_best_topk = 0.0
    total_risk_mae_primary = 0.0
    n = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        gt_box = batch["gt_box"].to(device, non_blocking=True)
        gt_risk = batch["gt_risk"].to(device, non_blocking=True)

        gt_boxes_topk = batch["gt_boxes_topk"].to(device, non_blocking=True)   # [B,K,4]
        gt_risks_topk = batch["gt_risks_topk"].to(device, non_blocking=True)   # [B,K]
        gt_mask_topk = batch["gt_mask_topk"].to(device, non_blocking=True)     # [B,K]

        out = model(pixel_values)
        if args.num_queries == 1:
            pred_box, pred_risk = out
            pred_boxes = pred_box[:, None, :]
            pred_risks = pred_risk[:, None]
        else:
            pred_boxes, pred_risks = out

        loss, loss_box, loss_risk = compute_loss_multiquery(
            pred_boxes=pred_boxes,
            pred_risks=pred_risks,
            gt_boxes_topk=gt_boxes_topk,
            gt_risks_topk=gt_risks_topk,
            gt_mask_topk=gt_mask_topk,
            args=args,
        )

        # --- primary prediction: query with max predicted risk ---
        primary_q = pred_risks.argmax(dim=1)  # [B]
        b_idx = torch.arange(pixel_values.shape[0], device=device)
        pred_primary_box = pred_boxes[b_idx, primary_q]
        pred_primary_risk = pred_risks[b_idx, primary_q]

        # IoU against top-1 GT (legacy) + best IoU against any GT in topk
        iou_primary = bbox_iou_xyxy(pred_primary_box, gt_box)

        # best IoU among topk GT (for diagnosing "wrong object but within topk")
        # pred_primary_box: [B,4], gt_boxes_topk: [B,K,4]
        B, K = gt_boxes_topk.shape[0], gt_boxes_topk.shape[1]
        pb = pred_primary_box[:, None, :].expand(B, K, 4)
        gb = gt_boxes_topk
        # flatten for iou
        iou_all = bbox_iou_xyxy(pb.reshape(-1, 4), gb.reshape(-1, 4)).reshape(B, K)
        iou_all = iou_all.masked_fill(~gt_mask_topk, -1.0)
        iou_best = iou_all.max(dim=1).values

        risk_mae = (pred_primary_risk - gt_risk).abs()

        bs = pixel_values.shape[0]
        total_loss += loss.item() * bs
        total_box += loss_box.item() * bs
        total_risk += loss_risk.item() * bs
        total_iou_primary += iou_primary.mean().item() * bs
        total_iou_best_topk += iou_best.clamp(min=0).mean().item() * bs
        total_risk_mae_primary += risk_mae.mean().item() * bs
        n += bs

    # DDP reduce
    t = torch.tensor(
        [total_loss, total_box, total_risk, total_iou_primary, total_iou_best_topk, total_risk_mae_primary, n],
        device=device,
    )
    t = ddp_reduce_sum(t)

    total_loss, total_box, total_risk, total_iou_primary, total_iou_best_topk, total_risk_mae_primary, n = t.tolist()
    n = max(1.0, float(n))

    return {
        "loss": total_loss / n,
        "loss_box": total_box / n,
        "loss_risk": total_risk / n,
        "iou_primary": total_iou_primary / n,
        "iou_best_topk": total_iou_best_topk / n,
        "risk_mae": total_risk_mae_primary / n,
        "num_samples": n,
    }


def build_loaders(args: argparse.Namespace, is_distributed: bool, rank: int, world_size: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = DramaFastDataset(
        jsonl_path=args.train_jsonl,
        data_root=args.data_root,
        num_frames=args.num_frames,
        stride=args.stride,
        img_size=args.img_size,
        topk_targets=args.topk_targets,
    )

    val_ds = DramaFastDataset(
        jsonl_path=args.val_jsonl,
        data_root=args.data_root,
        num_frames=args.num_frames,
        stride=args.stride,
        img_size=args.img_size,
        topk_targets=args.topk_targets,
    )

    train_sampler = None
    val_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=default_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=default_collate,
    )

    return train_loader, val_loader


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    # DDP init
    is_distributed = args.ddp and world_size > 1
    if is_distributed:
        os.environ.setdefault("MASTER_ADDR", args.master_addr)
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    # Seeds (each rank different)
    set_seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    # Data
    train_loader, val_loader = build_loaders(args, is_distributed, rank, world_size)

    # Model
    model = FastSystemPhase1(
        img_size=args.img_size,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        num_queries=args.num_queries,
        num_heads=args.head_num_heads,
        mlp_dim=args.head_mlp_dim,
    )
    model.to(device)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
            # find_unused_parameters=args.ddp_find_unused,  <-- 不要用 args 控制了，直接写死 True
            find_unused_parameters=True
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    state = TrainState(epoch=0, best_val_loss=float("inf"), global_step=0)
    if args.resume and Path(args.resume).exists():
        state = load_checkpoint(args.resume, model, optimizer, scaler=scaler)
        if is_main():
            print(f"[Resume] loaded {args.resume}: epoch={state.epoch} best_val_loss={state.best_val_loss:.6f}")

    # warmup steps / LR schedule
    global_step = int(getattr(state, "global_step", 0) or 0)
    if global_step <= 0 and int(getattr(state, "epoch", 0)) > 0:
        global_step = int(state.epoch) * max(1, len(train_loader))
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_schedule(step: int) -> float:
        # linear warmup then cosine decay
        if warmup_steps > 0 and step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

    if is_main():
        print("== Phase-1 Train ==")
        print(f"train={args.train_jsonl}\nval  ={args.val_jsonl}\ndata_root={args.data_root}")
        print(
            f"ddp={args.ddp} world_size={world_size} amp={args.amp} pretrained={args.pretrained} freeze_backbone={args.freeze_backbone} "
            f"num_queries={args.num_queries} topk_targets={args.topk_targets}"
        )

    # Init W&B (main process only)
    wandb = maybe_init_wandb(args, out_dir)

    for epoch in range(state.epoch, args.epochs):
        if is_distributed:
            assert isinstance(train_loader.sampler, DistributedSampler)
            train_loader.sampler.set_epoch(epoch)

        model.train()

        pbar = tqdm(train_loader, disable=not is_main(), desc=f"Epoch {epoch:03d}")
        running_loss = 0.0
        running_box = 0.0
        running_risk = 0.0
        seen = 0

        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)

            # fixed-shape topk supervision
            gt_boxes_topk = batch["gt_boxes_topk"].to(device, non_blocking=True)   # [B,K,4]
            gt_risks_topk = batch["gt_risks_topk"].to(device, non_blocking=True)   # [B,K]
            gt_mask_topk = batch["gt_mask_topk"].to(device, non_blocking=True)     # [B,K]

            # step lr
            lr = lr_schedule(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # autocast
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_cm = torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled())
            else:
                autocast_cm = torch.cuda.amp.autocast(enabled=scaler.is_enabled())

            with autocast_cm:
                out = model(pixel_values)
                if args.num_queries == 1:
                    pred_box, pred_risk = out
                    pred_boxes = pred_box[:, None, :]
                    pred_risks = pred_risk[:, None]
                else:
                    pred_boxes, pred_risks = out

                loss, loss_box, loss_risk = compute_loss_multiquery(
                    pred_boxes=pred_boxes,
                    pred_risks=pred_risks,
                    gt_boxes_topk=gt_boxes_topk,
                    gt_risks_topk=gt_risks_topk,
                    gt_mask_topk=gt_mask_topk,
                    args=args,
                )

                # grad accumulation
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (it + 1) % args.accum_steps == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = pixel_values.shape[0]
            running_loss += loss.item() * bs * args.accum_steps
            running_box += loss_box.item() * bs
            running_risk += loss_risk.item() * bs
            seen += bs
            global_step += 1

            if is_main():
                pbar.set_postfix(
                    loss=running_loss / max(1, seen),
                    box=running_box / max(1, seen),
                    risk=running_risk / max(1, seen),
                    lr=lr,
                )
                if wandb is not None and args.wandb_log_every > 0 and (global_step % args.wandb_log_every == 0):
                    wandb.log(
                        {
                            "train/loss": running_loss / max(1, seen),
                            "train/loss_box": running_box / max(1, seen),
                            "train/loss_risk": running_risk / max(1, seen),
                            "lr": lr,
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

        # Aggregate train stats (DDP)
        t = torch.tensor([running_loss, running_box, running_risk, seen], device=device)
        t = ddp_reduce_sum(t)
        tr_loss, tr_box, tr_risk, tr_n = t.tolist()
        tr_n = max(1.0, tr_n)
        train_stats = {
            "loss": tr_loss / tr_n,
            "loss_box": tr_box / tr_n,
            "loss_risk": tr_risk / tr_n,
            "num_samples": tr_n,
        }

        # Val
        val_stats = run_eval(model, val_loader, device, args)

        # Update resumable state before checkpointing
        state.epoch = epoch + 1
        state.global_step = int(global_step)

        # Save / log
        if is_main():
            record = {
                "epoch": epoch,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train": train_stats,
                "val": val_stats,
            }
            with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_stats['loss']:.4f} box={train_stats['loss_box']:.4f} risk={train_stats['loss_risk']:.4f} | "
                f"val loss={val_stats['loss']:.4f} iou_primary={val_stats['iou_primary']:.4f} iou_best_topk={val_stats['iou_best_topk']:.4f} risk_mae={val_stats['risk_mae']:.4f}"
            )

            if wandb is not None:
                wandb.log(
                    {
                        "epoch": int(epoch),
                        "train/loss_epoch": float(train_stats["loss"]),
                        "train/loss_box_epoch": float(train_stats["loss_box"]),
                        "train/loss_risk_epoch": float(train_stats["loss_risk"]),
                        "val/loss": float(val_stats["loss"]),
                        "val/iou_primary": float(val_stats["iou_primary"]),
                        "val/iou_best_topk": float(val_stats["iou_best_topk"]),
                        "val/risk_mae": float(val_stats["risk_mae"]),
                        "best/val_loss": float(min(state.best_val_loss, float(val_stats["loss"]))),
                    },
                    step=int(global_step),
                )

            save_checkpoint(out_dir, "last.pt", model, optimizer, state, args, scaler=scaler)
            if val_stats["loss"] < state.best_val_loss:
                state.best_val_loss = float(val_stats["loss"])
                save_checkpoint(out_dir, "best.pt", model, optimizer, state, args, scaler=scaler)

    if is_main():
        print("== Training Done ==")

    if is_distributed:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--data_root", required=True, help="DRAMA data root (DRAMA_DATA_ROOT)")

    # io
    p.add_argument("--out_dir", required=True)
    p.add_argument("--resume", default="")

    # dataset
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--topk_targets", type=int, default=1, help="K in top-k GT targets")
    p.add_argument("--verbose_every", type=int, default=0)

    # training
    p.add_argument("--batch_size", type=int, default=8, help="per-GPU batch size")
    p.add_argument("--eval_batch_size", type=int, default=0, help="0 => use batch_size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=2e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)

    # --- loss weights ---
    p.add_argument("--lambda_box", type=float, default=1.0)
    p.add_argument("--lambda_risk", type=float, default=2.0)
    p.add_argument("--box_l1_weight", type=float, default=5.0, help="DETR-style L1 weight")
    p.add_argument("--box_giou_weight", type=float, default=2.0, help="DETR-style GIoU weight")
    p.add_argument("--match_risk_weight", type=float, default=1.0, help="risk term in matching cost")
    p.add_argument("--risk_noobj_weight", type=float, default=0.25, help="downweight unmatched query risk loss")

    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")

    # wandb
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (main process only)")
    p.add_argument("--wandb_entity", type=str, default="haozhuangchi")
    p.add_argument("--wandb_project", type=str, default="aumovio")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_log_every", type=int, default=50, help="log train metrics every N steps")

    # model
    p.add_argument("--pretrained", action="store_true", help="use pretrained Swin3D weights")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--num_queries", type=int, default=1, help="number of learnable queries")
    p.add_argument("--head_num_heads", type=int, default=8)
    p.add_argument("--head_mlp_dim", type=int, default=256)

    # ddp
    p.add_argument("--ddp", action="store_true", help="spawn DDP workers inside this script")
    p.add_argument("--gpus", type=int, default=1, help="number of local GPUs to use when --ddp")
    p.add_argument("--master_addr", default="127.0.0.1")
    p.add_argument("--master_port", type=int, default=29501)
    p.add_argument("--ddp_find_unused", action="store_true", help="use find_unused_parameters=True (safe but slower)")

    # misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.eval_batch_size == 0:
        args.eval_batch_size = None

    if args.ddp:
        world_size = int(args.gpus)
        assert world_size >= 1
        torch.multiprocessing.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
