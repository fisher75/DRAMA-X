"""Phase-1 training (Single-Query): Video Swin-T backbone + single-query joint head.

This is a *rollback* single-query trainer with a stronger box loss and safer IoU:
- ✅ Valid-box fix: already enforced inside `JointQueryHead` (x1<x2, y1<y2).
- ✅ Box loss: SmoothL1 (or L1) + (1 - GIoU) (DETR-style), both on normalized xyxy.
- ✅ Metrics: IoU on the primary target + optional best-IoU over top-k targets.

Why this file exists:
- Your multi-query + matching branch can be powerful, but it's easy to destabilize early
  training when supervision is noisy / multi-target counts vary.
- This script keeps the already-working single-query interface and upgrades only the loss.

Run (single GPU):
  python -m drama_fast.train.train_phase1_single_giou --train_jsonl ... --val_jsonl ... --data_root ... --out_dir ...

Run (DDP):
  python -m drama_fast.train.train_phase1_single_giou --ddp --gpus 4 ...

命令：
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DRAMA_DATA_ROOT=/data2/automan/data/drama_data

python -m drama_fast.train.train_phase1_single_giou \
  --train_jsonl ./splits_v3/train.jsonl \
  --val_jsonl   ./splits_v3/val.jsonl \
  --data_root   $DRAMA_DATA_ROOT \
  --out_dir     ./runs/phase1_swin_t_v3_single_giou \
  --num_frames 8 --img_size 224 --stride 2 \
  --topk_targets 5 \
  --batch_size 16 \
  --epochs 50 \
  --lr 2e-4 --weight_decay 0.05 \
  --lambda_box 1.0 --lambda_risk 2.0 \
  --box_l1_weight 1.0 --box_giou_weight 2.0 \
  --use_smooth_l1 --smooth_l1_beta 0.1 \
  --amp \
  --pretrained \
  --ddp --gpus 4 \
  --wandb \
  --wandb_run_name "SwinT-SingleQuery-GIoU-SmoothL1-v3"

384版本：
# 只有当你跑通了 224 版本，觉得效果还不够好时，再跑这个
python -m drama_fast.train.train_phase1_single_giou \
  --train_jsonl ./splits_v3/train.jsonl \
  --val_jsonl   ./splits_v3/val.jsonl \
  --data_root   $DRAMA_DATA_ROOT \
  --out_dir     ./runs/phase1_swin_t_v3_single_giou_384 \
  --num_frames 8 --img_size 384 --stride 2 \
  --topk_targets 5 \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-4 --weight_decay 0.05 \
  --lambda_box 1.0 --lambda_risk 2.0 \
  --box_l1_weight 1.0 --box_giou_weight 2.0 \
  --use_smooth_l1 --smooth_l1_beta 0.1 \
  --amp \
  --pretrained \
  --ddp --gpus 4 \
  --wandb \
  --wandb_run_name "SwinT-Single-GIoU-SmoothL1-v3-384"
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

from drama_fast.dataset_phase1 import DramaFastDataset
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


def _box_area_xyxy(box: torch.Tensor) -> torch.Tensor:
    # box: [...,4]
    x1, y1, x2, y2 = box.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU for normalized xyxy boxes.

    a, b: [...,4] in xyxy, each coord in [0,1]
    returns: [...]
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

    a, b: [...,4] in xyxy, each coord in [0,1]
    returns: [...]
    """
    # IoU
    iou = bbox_iou_xyxy(a, b, eps=eps)

    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    # smallest enclosing box
    cx1 = torch.min(ax1, bx1)
    cy1 = torch.min(ay1, by1)
    cx2 = torch.max(ax2, bx2)
    cy2 = torch.max(ay2, by2)

    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)

    # union = area_a + area_b - inter
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    # inter = iou * union  => union = (area_a+area_b) / (1+iou) ??? not stable
    # compute inter directly again for numerical stability
    ix1 = torch.max(ax1, bx1)
    iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2)
    iy2 = torch.min(ay2, by2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    union = area_a + area_b - inter

    giou = iou - (c_area - union) / (c_area + eps)
    return giou


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


def _best_iou_over_topk(pred_box: torch.Tensor, gt_boxes_topk_list) -> torch.Tensor:
    """pred_box: [B,4], gt_boxes_topk_list: list[tensor[K,4]] (K can vary)
    returns: [B]
    """
    b = pred_box.shape[0]
    out = torch.zeros((b,), device=pred_box.device, dtype=pred_box.dtype)
    for i in range(b):
        gtb = gt_boxes_topk_list[i]
        if gtb is None or (isinstance(gtb, torch.Tensor) and gtb.numel() == 0):
            out[i] = 0.0
            continue
        if not isinstance(gtb, torch.Tensor):
            gtb = torch.tensor(gtb, device=pred_box.device, dtype=pred_box.dtype)
        gtb = gtb.to(device=pred_box.device, dtype=pred_box.dtype)
        pb = pred_box[i].unsqueeze(0).expand(gtb.shape[0], -1)
        out[i] = bbox_iou_xyxy(pb, gtb).max()
    return out


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_box: float,
    lambda_risk: float,
    box_l1_weight: float,
    box_giou_weight: float,
    use_smooth_l1: bool,
    smooth_l1_beta: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_box = 0.0
    total_box_l1 = 0.0
    total_box_giou = 0.0
    total_risk = 0.0
    total_iou_primary = 0.0
    total_iou_best_topk = 0.0
    total_risk_mae = 0.0
    n = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        gt_box = batch["gt_box"].to(device, non_blocking=True)
        gt_risk = batch["gt_risk"].to(device, non_blocking=True)
        gt_boxes_topk = batch.get("gt_boxes_topk", None)

        pred_box, pred_risk = model(pixel_values)

        if use_smooth_l1:
            loss_l1 = F.smooth_l1_loss(pred_box, gt_box, beta=smooth_l1_beta)
        else:
            loss_l1 = F.l1_loss(pred_box, gt_box)
        loss_giou = (1.0 - bbox_giou_xyxy(pred_box, gt_box)).mean()
        loss_box = box_l1_weight * loss_l1 + box_giou_weight * loss_giou

        loss_risk = F.mse_loss(pred_risk, gt_risk)
        loss = lambda_box * loss_box + lambda_risk * loss_risk

        iou_primary = bbox_iou_xyxy(pred_box, gt_box).mean()
        if gt_boxes_topk is not None:
            iou_best = _best_iou_over_topk(pred_box, gt_boxes_topk).mean()
        else:
            iou_best = iou_primary

        risk_mae = (pred_risk - gt_risk).abs().mean()

        bs = pixel_values.shape[0]
        total_loss += loss.item() * bs
        total_box += loss_box.item() * bs
        total_box_l1 += loss_l1.item() * bs
        total_box_giou += loss_giou.item() * bs
        total_risk += loss_risk.item() * bs
        total_iou_primary += iou_primary.item() * bs
        total_iou_best_topk += iou_best.item() * bs
        total_risk_mae += risk_mae.item() * bs
        n += bs

    # DDP reduce
    t = torch.tensor(
        [
            total_loss,
            total_box,
            total_box_l1,
            total_box_giou,
            total_risk,
            total_iou_primary,
            total_iou_best_topk,
            total_risk_mae,
            n,
        ],
        device=device,
    )
    t = ddp_reduce_sum(t)
    (
        total_loss,
        total_box,
        total_box_l1,
        total_box_giou,
        total_risk,
        total_iou_primary,
        total_iou_best_topk,
        total_risk_mae,
        n,
    ) = t.tolist()
    n = max(1.0, float(n))

    return {
        "loss": total_loss / n,
        "loss_box": total_box / n,
        "loss_box_l1": total_box_l1 / n,
        "loss_box_giou": total_box_giou / n,
        "loss_risk": total_risk / n,
        "iou_primary": total_iou_primary / n,
        "iou_best_topk": total_iou_best_topk / n,
        "risk_mae": total_risk_mae / n,
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
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
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
    model = FastSystemPhase1(img_size=args.img_size, pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    model.to(device)

    if is_distributed:
        # 加上 find_unused_parameters=True 解决 VideoSwin 未使用层报错
        model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    state = TrainState(epoch=0, best_val_loss=float("inf"), global_step=0)
    if args.resume and Path(args.resume).exists():
        state = load_checkpoint(args.resume, model, optimizer, scaler=scaler)
        if is_main():
            print(f"[Resume] loaded {args.resume}: epoch={state.epoch} best_val_loss={state.best_val_loss:.6f}")

    # Training weights
    lambda_box = args.lambda_box
    lambda_risk = args.lambda_risk

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
        print("== Phase-1 Train (Single-Query + GIoU) ==")
        print(f"train={args.train_jsonl}\nval  ={args.val_jsonl}\ndata_root={args.data_root}")
        print(
            f"ddp={args.ddp} world_size={world_size} amp={args.amp} pretrained={args.pretrained} freeze_backbone={args.freeze_backbone}"
        )
        print(
            f"box_loss: {'smoothl1' if args.use_smooth_l1 else 'l1'} (w_l1={args.box_l1_weight}, w_giou={args.box_giou_weight}), "
            f"lambda_box={lambda_box}, lambda_risk={lambda_risk}"
        )

    # Init W&B (main process only)
    wandb = maybe_init_wandb(args, out_dir)

    for epoch in range(state.epoch, args.epochs):
        # DDP epoch seed
        if is_distributed:
            assert isinstance(train_loader.sampler, DistributedSampler)
            train_loader.sampler.set_epoch(epoch)

        model.train()

        pbar = tqdm(train_loader, disable=not is_main(), desc=f"Epoch {epoch:03d}")
        running_loss = 0.0
        running_box = 0.0
        running_box_l1 = 0.0
        running_box_giou = 0.0
        running_risk = 0.0
        seen = 0

        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            gt_box = batch["gt_box"].to(device, non_blocking=True)
            gt_risk = batch["gt_risk"].to(device, non_blocking=True)

            # step lr
            lr = lr_schedule(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_cm = torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled())
            else:
                autocast_cm = torch.cuda.amp.autocast(enabled=scaler.is_enabled())

            with autocast_cm:
                pred_box, pred_risk = model(pixel_values)

                if args.use_smooth_l1:
                    loss_l1 = F.smooth_l1_loss(pred_box, gt_box, beta=args.smooth_l1_beta)
                else:
                    loss_l1 = F.l1_loss(pred_box, gt_box)
                loss_giou = (1.0 - bbox_giou_xyxy(pred_box, gt_box)).mean()
                loss_box = args.box_l1_weight * loss_l1 + args.box_giou_weight * loss_giou

                loss_risk = F.mse_loss(pred_risk, gt_risk)
                loss = lambda_box * loss_box + lambda_risk * loss_risk

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
            running_box_l1 += loss_l1.item() * bs
            running_box_giou += loss_giou.item() * bs
            running_risk += loss_risk.item() * bs
            seen += bs
            global_step += 1

            if is_main():
                pbar.set_postfix(
                    loss=running_loss / max(1, seen),
                    box=running_box / max(1, seen),
                    l1=running_box_l1 / max(1, seen),
                    giou=running_box_giou / max(1, seen),
                    risk=running_risk / max(1, seen),
                    lr=lr,
                )
                if wandb is not None and args.wandb_log_every > 0 and (global_step % args.wandb_log_every == 0):
                    wandb.log(
                        {
                            "train/loss": running_loss / max(1, seen),
                            "train/loss_box": running_box / max(1, seen),
                            "train/loss_box_l1": running_box_l1 / max(1, seen),
                            "train/loss_box_giou": running_box_giou / max(1, seen),
                            "train/loss_risk": running_risk / max(1, seen),
                            "lr": lr,
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

        # Aggregate train stats (DDP)
        t = torch.tensor([running_loss, running_box, running_box_l1, running_box_giou, running_risk, seen], device=device)
        t = ddp_reduce_sum(t)
        tr_loss, tr_box, tr_box_l1, tr_box_giou, tr_risk, tr_n = t.tolist()
        tr_n = max(1.0, tr_n)
        train_stats = {
            "loss": tr_loss / tr_n,
            "loss_box": tr_box / tr_n,
            "loss_box_l1": tr_box_l1 / tr_n,
            "loss_box_giou": tr_box_giou / tr_n,
            "loss_risk": tr_risk / tr_n,
            "num_samples": tr_n,
        }

        # Val
        val_stats = run_eval(
            model,
            val_loader,
            device,
            lambda_box=lambda_box,
            lambda_risk=lambda_risk,
            box_l1_weight=args.box_l1_weight,
            box_giou_weight=args.box_giou_weight,
            use_smooth_l1=args.use_smooth_l1,
            smooth_l1_beta=args.smooth_l1_beta,
        )

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
                f"train loss={train_stats['loss']:.4f} box={train_stats['loss_box']:.4f} (l1={train_stats['loss_box_l1']:.4f}, giou={train_stats['loss_box_giou']:.4f}) risk={train_stats['loss_risk']:.4f} | "
                f"val loss={val_stats['loss']:.4f} iou_primary={val_stats['iou_primary']:.4f} iou_best_topk={val_stats['iou_best_topk']:.4f} risk_mae={val_stats['risk_mae']:.4f}"
            )

            if wandb is not None:
                wandb.log(
                    {
                        "epoch": int(epoch),
                        "train/loss_epoch": float(train_stats["loss"]),
                        "train/loss_box_epoch": float(train_stats["loss_box"]),
                        "train/loss_box_l1_epoch": float(train_stats["loss_box_l1"]),
                        "train/loss_box_giou_epoch": float(train_stats["loss_box_giou"]),
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
    p.add_argument("--topk_targets", type=int, default=1, help="keep top-k targets in dataset meta (primary target is always #0)")

    # training
    p.add_argument("--batch_size", type=int, default=8, help="per-GPU batch size")
    p.add_argument("--eval_batch_size", type=int, default=0, help="0 => use batch_size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=2e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lambda_box", type=float, default=1.0)
    p.add_argument("--lambda_risk", type=float, default=2.0)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")

    # box loss
    p.add_argument("--use_smooth_l1", action="store_true", help="use SmoothL1 for box regression (recommended)")
    p.add_argument("--smooth_l1_beta", type=float, default=0.1)
    p.add_argument("--box_l1_weight", type=float, default=1.0)
    p.add_argument("--box_giou_weight", type=float, default=2.0)

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

    # ddp
    p.add_argument("--ddp", action="store_true", help="spawn DDP workers inside this script")
    p.add_argument("--gpus", type=int, default=1, help="number of local GPUs to use when --ddp")
    p.add_argument("--master_addr", default="127.0.0.1")
    p.add_argument("--master_port", type=int, default=29501)

    # misc
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    # default SmoothL1 on unless user explicitly turns it off
    # (keeps CLI compatibility: you can disable via --use_smooth_l1 false by not passing it and setting beta large)
    if not args.use_smooth_l1:
        # keep legacy behavior: L1
        pass
    return args


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
