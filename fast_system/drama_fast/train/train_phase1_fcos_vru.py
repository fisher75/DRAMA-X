"""
# 建议指定 CUDA 设备，防止抢占
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# 运行训练
torchrun --nproc_per_node=4 -m drama_fast.train.train_phase1_fcos_vru \
  --ddp \
  --train_jsonl ./splits_v3/train.jsonl \
  --val_jsonl   ./splits_v3/val.jsonl \
  --data_root   $DRAMA_DATA_ROOT \
  --out_dir     ./runs/phase1_fcos_vru_swin_t_384 \
  --batch_size 8 \
  --epochs 100 \
  --img_size 384 \
  --num_frames 8 \
  --topk_targets 5 \
  --pretrained \
  --lr 1e-4 \
  --warmup_ratio 0.05 \
  --wandb \
  --wandb_run_name "SwinT-FCOS-VRU-384-Warmup"
"""

import argparse
import json
import os
import random
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from drama_fast.dataset_phase1 import DramaFastDataset as DatasetPhase1
from drama_fast.models.model_phase1_fcos import FCOSConfig, FCOSLoss, Phase1VideoSwinFCOS


# ------------------------
# Utils
# ------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def maybe_init_wandb(args: argparse.Namespace, out_dir: Path):
    """Init Weights & Biases on main process only. Safe for torchrun/DDP."""
    if not getattr(args, "wandb", False):
        return None

    # Disable W&B in non-main processes to avoid duplicate runs.
    if not is_main_process():
        os.environ.setdefault("WANDB_MODE", "disabled")
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[WARN] W&B enabled but import failed: {e}")
        return None

    run_name = getattr(args, "wandb_run_name", "") or None
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

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_setup(local_rank: int):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")


def ddp_cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def reduce_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """All-reduce a dict of scalar tensors."""
    if not is_dist_avail_and_initialized():
        return {k: float(v.item()) for k, v in input_dict.items()}

    keys = sorted(input_dict.keys())
    vals = torch.stack([input_dict[k] for k in keys])
    dist.all_reduce(vals)
    vals /= dist.get_world_size()
    return {k: float(vals[i].item()) for i, k in enumerate(keys)}


# ------------------------
# Metrics
# ------------------------

def compute_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU for aligned boxes. boxes*: [N,4]"""
    x11, y11, x12, y12 = boxes1.unbind(-1)
    x21, y21, x22, y22 = boxes2.unbind(-1)

    xi1 = torch.max(x11, x21)
    yi1 = torch.max(y11, y21)
    xi2 = torch.min(x12, x22)
    yi2 = torch.min(y12, y22)

    inter = (xi2 - xi1).clamp(min=0) * (yi2 - yi1).clamp(min=0)
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1 + area2 - inter + 1e-6
    return inter / union


@torch.no_grad()
def eval_iou(
    model: Phase1VideoSwinFCOS,
    dl: DataLoader,
    device: torch.device,
    img_size: int,
    score_thr: float = 0.05,
    nms_thr: float = 0.6,
    topk: int = 300,
) -> Dict[str, float]:
    """A simple quality check:
    - iou_primary: best predicted box IoU w.r.t. the primary GT box
    - iou_best_topk: best predicted box IoU w.r.t. any GT in topk list
    """
    model.eval()

    iou_primary_sum = 0.0
    iou_best_sum = 0.0
    n_samples = 0

    for batch in dl:
        clip = batch["pixel_values"].to(device, non_blocking=True)           # [B,T,3,H,W] (after our dataset fix)
        gt_box = batch["gt_box"].to(device, non_blocking=True)               # [B,4] normalized
        gt_boxes_topk = batch["gt_boxes_topk"].to(device, non_blocking=True) # [B,K,4] normalized
        gt_mask_topk = batch["gt_mask_topk"].to(device, non_blocking=True)   # [B,K] bool

        B = clip.shape[0]
        preds = model.inference(clip, img_hw=(img_size, img_size), score_thr=score_thr, nms_thr=nms_thr, topk=topk)

        # convert gt to pixel
        gt_primary_px = gt_box * img_size
        gt_topk_px = gt_boxes_topk * img_size

        for b in range(B):
            pred_boxes = preds[b]["boxes"]
            if pred_boxes.numel() == 0:
                iou_primary = 0.0
                iou_best = 0.0
            else:
                pb = pred_boxes
                g0 = gt_primary_px[b].unsqueeze(0)
                ious0 = compute_iou_xyxy(pb, g0.expand(pb.shape[0], 4))
                iou_primary = float(ious0.max().item())

                valid = gt_mask_topk[b]
                gk = gt_topk_px[b][valid]
                if gk.numel() == 0:
                    iou_best = iou_primary
                else:
                    best = 0.0
                    for j in range(gk.shape[0]):
                        ious = compute_iou_xyxy(pb, gk[j].unsqueeze(0).expand(pb.shape[0], 4))
                        best = max(best, float(ious.max().item()))
                    iou_best = best

            iou_primary_sum += iou_primary
            iou_best_sum += iou_best
            n_samples += 1

    return {
        "iou_primary": iou_primary_sum / max(1, n_samples),
        "iou_best_topk": iou_best_sum / max(1, n_samples),
        "num_samples": float(n_samples),
    }


# ------------------------
# Main
# ------------------------

def parse_args():
    p = argparse.ArgumentParser("Phase1 FCOS VRU detector (dense head)")

    # data
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--topk_targets", type=int, default=5)

    # model
    p.add_argument("--num_classes", type=int, default=1, help="Default=1 (VRU). Set to 2 only if your dataset provides per-box class labels.")

    # train
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--pretrained", action="store_true")

    # ddp
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    # eval decode
    p.add_argument("--score_thr", type=float, default=0.05)
    p.add_argument("--nms_thr", type=float, default=0.6)
    p.add_argument("--topk", type=int, default=300)
    
    # wandb
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (main process only)")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="aumovio")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_log_every", type=int, default=50, help="log train metrics every N steps")
    
    # model
    p.add_argument("--backbone", type=str, default="swin_t", help="backbone name (e.g., swin_t)")
    
    # [FIX] Add scheduler args to ArgumentParser
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.05)


    return p.parse_args()


def main():
    args = parse_args()

    if args.ddp:
        ddp_setup(args.local_rank)

    wandb = None
    global_step = 0  # ✅ 所有进程都定义，避免非主进程 UnboundLocalError

    try:
        device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
        seed_everything(args.seed + get_rank())

        out_dir = Path(args.out_dir)

        # ✅ 先由主进程创建目录/写参数
        if is_main_process():
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "ckpts").mkdir(exist_ok=True)
            (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

        # ✅ 等目录准备好后再 barrier（可选但推荐）
        if args.ddp:
            dist.barrier()

        # ✅ 只 init 一次
        wandb = maybe_init_wandb(args, out_dir)                # ✅ 建议加：全局 step


        # Build config
        cfg = FCOSConfig(num_classes=args.num_classes)
        if is_main_process():
            (out_dir / "fcos_cfg.json").write_text(json.dumps(asdict(cfg), indent=2))

        # Datasets
        train_ds = DatasetPhase1(
            jsonl_path=args.train_jsonl,
            data_root=args.data_root,
            num_frames=args.num_frames,
            img_size=args.img_size,
            stride=args.stride,
            topk_targets=args.topk_targets,
            return_meta=False,
        )
        val_ds = DatasetPhase1(
            jsonl_path=args.val_jsonl,
            data_root=args.data_root,
            num_frames=args.num_frames,
            img_size=args.img_size,
            stride=args.stride,
            topk_targets=args.topk_targets,
            return_meta=False,
        )

        train_sampler = DistributedSampler(train_ds, shuffle=True) if args.ddp else None
        # val_sampler = DistributedSampler(val_ds, shuffle=False) if args.ddp else None
        val_sampler = None

        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        # Model + loss
        # model = Phase1VideoSwinFCOS(cfg=cfg, pretrained=args.pretrained)
        model = Phase1VideoSwinFCOS(cfg=cfg, backbone_name=args.backbone, pretrained=args.pretrained)
        model.to(device)
        loss_fn = FCOSLoss(cfg=cfg)

        if args.ddp:
            model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp) # 原有代码

        # 计算总步数
        total_steps = len(train_dl) * args.epochs
        warmup_steps = int(args.warmup_ratio * total_steps) # 5% 的步数用于热身

        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(current_step: int):
            # 1. Warmup 阶段：线性增加
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # 2. Cosine Decay 阶段：余弦衰减
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        best_iou = -1.0

        for epoch in range(args.epochs):
            if args.ddp:
                train_sampler.set_epoch(epoch)

            model.train()
            loss_sum = 0.0
            n_steps = 0

            for batch in train_dl:
                clip = batch["pixel_values"].to(device, non_blocking=True)           # [B,T,3,H,W]
                gt_boxes_topk = batch["gt_boxes_topk"].to(device, non_blocking=True) # [B,K,4] normalized
                gt_mask_topk = batch["gt_mask_topk"].to(device, non_blocking=True)   # [B,K] bool

                # Build per-image GT lists (pixel xyxy)
                B = clip.shape[0]
                gt_boxes_list: List[torch.Tensor] = []
                gt_labels_list: List[torch.Tensor] = []
                for b in range(B):
                    valid = gt_mask_topk[b]
                    boxes_px = gt_boxes_topk[b][valid] * args.img_size
                    gt_boxes_list.append(boxes_px)
                    # default: 1-class VRU => label 0
                    gt_labels_list.append(torch.zeros((boxes_px.shape[0],), device=device, dtype=torch.long))

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=args.amp):
                    outputs = model(clip)
                    loss_dict = loss_fn(outputs, gt_boxes_list, gt_labels_list, img_hw=(args.img_size, args.img_size))
                    loss = loss_dict["loss"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step() # <--- [新增] 每个 step 更新一次学习率

                loss_sum += float(loss.item())
                n_steps += 1
                # ---- wandb step log (main process only) ----
                if is_main_process():
                    global_step += 1
                    if (wandb is not None) and (args.wandb_log_every > 0) and (global_step % args.wandb_log_every == 0):
                        wandb.log(
                            {
                                "train/loss_step": float(loss.item()),
                                "train/epoch": epoch,
                                "train/step_in_epoch": n_steps,
                                "lr": float(optimizer.param_groups[0]["lr"]),
                            },
                            step=global_step,
                        )

            train_loss = torch.tensor(loss_sum / max(1, n_steps), device=device)
            train_loss_red = reduce_dict({"train_loss": train_loss})["train_loss"]

            if args.ddp:
                dist.barrier()

            # ---- Val: only run on main to avoid DDP aggregation complexity ----
            if is_main_process():
                metrics = eval_iou(
                    model.module if isinstance(model, DDP) else model,
                    val_dl,
                    device,
                    img_size=args.img_size,
                    score_thr=args.score_thr,
                    nms_thr=args.nms_thr,
                    topk=args.topk,
                )
            else:
                metrics = {"iou_primary": 0.0, "iou_best_topk": 0.0, "num_samples": 0.0}

            if args.ddp:
                dist.barrier()

            if is_main_process():
                print(
                    f"Epoch {epoch:03d} | train loss={train_loss_red:.4f} | "
                    f"val iou_primary={metrics['iou_primary']:.4f} iou_best_topk={metrics['iou_best_topk']:.4f}"
                )

                if wandb is not None:
                    wandb.log(
                        {
                            "train/loss_epoch": float(train_loss_red),
                            "val/iou_primary": float(metrics["iou_primary"]),
                            "val/iou_best_topk": float(metrics["iou_best_topk"]),
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )


                ckpt = {
                    "epoch": epoch,
                    "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                    "cfg": asdict(cfg),
                }
                torch.save(ckpt, out_dir / "ckpts" / "last.pt")

                cur = float(metrics["iou_best_topk"])
                if cur > best_iou:
                    best_iou = cur
                    torch.save(ckpt, out_dir / "ckpts" / "best.pt")

            if args.ddp:
                dist.barrier()

    finally:
        if is_main_process():
            try:
                if "wandb" in locals() and wandb is not None:
                    wandb.finish()
            except Exception:
                pass
        ddp_cleanup()


if __name__ == "__main__":
    main()
