"""Train Stage-2: proposal scoring / risk selection (two-stage pipeline).

Stage-1 produces proposal boxes for each keyframe (offline, e.g., via YOLO/ByteTrack).
Stage-2 learns to rank these proposals so that the GT VRU box is retrieved in Top-K.

This training is **NOT** end-to-end object detection. There is no box regression.
We only learn a scoring function over proposal boxes.

Outputs (in --out_dir):
- ckpt_last.pt / ckpt_best.pt
- config_stage2.json
- metrics_last.json
- (optional) debug_bad_samples.json (when evaluation detects inconsistencies)

W&B logging:
- step-level: loss, lr, pos_ratio, valid_count, grad_norm, steps_per_sec
- epoch-level: recall@K, stage1 ceiling, IoU stats
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from drama_fast.dataset_stage2 import Stage2RiskDataset, stage2_collate
from drama_fast.models.model_stage2_risk_transformer import Stage2RiskSelector
from drama_fast.train.eval_stage2 import evaluate_stage2


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dist_info() -> Dict[str, Any]:
    """Return distributed training info inferred from env vars."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_main = (rank == 0)
    return {
        "is_distributed": is_distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": is_main,
    }


def _dist_init(local_rank: int) -> None:
    """Init torch.distributed if launched by torchrun."""
    if dist.is_available() and not dist.is_initialized():
        # torchrun sets MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK
        dist.init_process_group(backend="nccl", init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def _dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _masked_bce_with_dynamic_pos_weight(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, Dict[str, float]]:
    """BCEWithLogits over valid proposals with a dynamic pos_weight.

    logits:  [B, N]
    targets: [B, N] in {0,1}
    mask:    [B, N] bool
    """
    # ensure tensors, not dict-like
    if isinstance(logits, dict):
        logits = logits["logits"]

    valid_logits = logits[mask]
    valid_targets = targets[mask]

    stats = {
        "valid_count": float(valid_targets.numel()),
        "pos_count": float(valid_targets.sum().item()),
    }
    stats["pos_ratio"] = float(stats["pos_count"] / max(stats["valid_count"], 1.0))

    if valid_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device), stats

    pos = float(valid_targets.sum().item())
    neg = float(valid_targets.numel()) - pos

    # If no positives in this batch, fall back to standard BCE.
    if pos <= 0.0:
        loss = nn.functional.binary_cross_entropy_with_logits(valid_logits, valid_targets)
        stats["pos_weight"] = 1.0
        return loss, stats

    pos_weight = max(neg / pos, 1.0)
    stats["pos_weight"] = float(pos_weight)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=logits.device))
    return loss_fn(valid_logits, valid_targets), stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--train_label_jsonl", type=str, required=True)
    p.add_argument("--train_proposals_jsonl", type=str, required=True)
    p.add_argument("--val_label_jsonl", type=str, required=True)
    p.add_argument("--val_proposals_jsonl", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)

    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--frame_stride", type=int, default=1)

    p.add_argument("--topk_proposals", type=int, default=80)
    p.add_argument("--iou_pos_thr", type=float, default=0.5)

    p.add_argument("--use_letterbox", action="store_true")
    p.add_argument("--letterbox_fill", type=int, default=114)

    p.add_argument("--pretrained_backbone", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    p.add_argument("--workers_per_gpu", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="dramax")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--run_name", dest="wandb_run_name", type=str, default=None)

    p.add_argument("--log_every", type=int, default=50, help="Print log every N optimizer steps")
    p.add_argument("--max_steps", type=int, default=-1, help="Stop after this many optimizer steps (<=0 means no limit)")

    # evaluation knobs
    p.add_argument("--eval_topk", type=str, default="1,5,10", help="Comma-separated K for recall@K")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------
    # DDP / multi-GPU support
    # ------------------------------
    # Use `torchrun --nproc_per_node=4 ...`.
    # We auto-enable DDP when WORLD_SIZE>1.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_main = rank == 0

    if is_distributed:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    # Seed: make each rank different but reproducible
    _set_seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # parse eval ks
    ks = tuple(int(x) for x in args.eval_topk.split(",") if x.strip())
    if not ks:
        ks = (1, 5, 10)

    if is_main:
        print("[stage2] starting Stage-2 risk training")
        if is_distributed:
            print(f"[stage2] distributed=ON | world_size={world_size} | rank={rank} | local_rank={local_rank}")
        else:
            print("[stage2] distributed=OFF")
        print(f"[stage2] device={device} | out_dir={args.out_dir}")
        print(f"[stage2] train_label_jsonl={args.train_label_jsonl}")
        print(f"[stage2] train_proposals_jsonl={args.train_proposals_jsonl}")
        print(f"[stage2] val_label_jsonl={args.val_label_jsonl}")
        print(f"[stage2] val_proposals_jsonl={args.val_proposals_jsonl}")

    # Datasets
    train_ds = Stage2RiskDataset(
        label_jsonl=args.train_label_jsonl,
        proposals_jsonl=args.train_proposals_jsonl,
        data_root=args.data_root,
        img_size=args.img_size,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        topk_proposals=args.topk_proposals,
        use_letterbox=args.use_letterbox,
        letterbox_fill=args.letterbox_fill,
        iou_pos_thr=args.iou_pos_thr,
        return_meta=False,
    )
    val_ds = Stage2RiskDataset(
        label_jsonl=args.val_label_jsonl,
        proposals_jsonl=args.val_proposals_jsonl,
        data_root=args.data_root,
        img_size=args.img_size,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        topk_proposals=args.topk_proposals,
        use_letterbox=args.use_letterbox,
        letterbox_fill=args.letterbox_fill,
        iou_pos_thr=args.iou_pos_thr,
        return_meta=True,  # enable sample_id for debug
    )

    train_sampler = None
    if is_distributed:
        # Each GPU sees a disjoint shard.
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers_per_gpu,
        pin_memory=True,
        collate_fn=stage2_collate,
    )
    # Keep validation on rank-0 only (simple + deterministic).
    # This avoids cross-rank metric reduction complexity.
    val_loader = None
    if is_main:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.workers_per_gpu // 2),
            pin_memory=True,
            collate_fn=stage2_collate,
        )

    if is_main:
        print(f"[stage2] train_samples={len(train_ds)} val_samples={len(val_ds)}")
        print(f"[stage2] train_batches_per_epoch(per-rank)={len(train_loader)} val_batches={len(val_loader)}")

    # Model
    model = Stage2RiskSelector(
        pretrained_backbone=args.pretrained_backbone,
        max_tokens=args.topk_proposals,
    )
    if args.freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # warmup + cosine schedule
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # Optional W&B
    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    best_key = "recall@5" if 5 in ks else f"recall@{ks[0]}"
    best_val = -1.0
    global_step = 0

    _save_json(str(out_dir / "config_stage2.json"), vars(args))

    for epoch in range(1, args.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"[stage2] epoch {epoch}/{args.epochs} ...")
        model.train()

        running_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for batch in train_loader:
            step_t0 = time.time()

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            proposals = batch["proposals"].to(device, non_blocking=True)
            proposal_mask = batch["proposal_mask"].to(device, non_blocking=True)
            proposal_conf = batch.get("proposal_conf")
            if proposal_conf is not None:
                proposal_conf = proposal_conf.to(device, non_blocking=True)
            gt_mask = batch["gt_mask"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            out = model(
                pixel_values=pixel_values,
                proposals=proposals,
                proposal_mask=proposal_mask,
                proposal_conf=proposal_conf,
            )
            logits = out["logits"] if isinstance(out, dict) else out
            targets = gt_mask.float()

            loss, loss_stats = _masked_bce_with_dynamic_pos_weight(logits, targets, proposal_mask)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
            optim.step()
            scheduler.step()

            running_loss += float(loss.item())
            n_steps += 1

            # step log
            steps_per_sec = 1.0 / max(time.time() - step_t0, 1e-6)
            if (global_step % max(1, args.log_every)) == 0:
                lr = float(optim.param_groups[0]["lr"])
                pos = int(loss_stats["pos_count"])
                valid = int(loss_stats["valid_count"])
                pr = 100.0 * float(loss_stats["pos_ratio"])
                print(
                    f"  [e{epoch:02d} step{global_step:06d}] "
                    f"loss={loss.item():.4f} lr={lr:.2e} "
                    f"pos={pos}/{valid} ({pr:.1f}%) "
                    f"{steps_per_sec:.2f} steps/s"
                )

            if wb is not None:
                wb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/lr": float(optim.param_groups[0]["lr"]),
                        "train/pos_ratio": float(loss_stats["pos_ratio"]),
                        "train/valid_count": float(loss_stats["valid_count"]),
                        "train/pos_weight": float(loss_stats.get("pos_weight", 1.0)),
                        "train/grad_norm": float(grad_norm),
                        "train/steps_per_sec": float(steps_per_sec),
                    },
                    step=global_step,
                )

            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        # Aggregate epoch loss across GPUs for consistent reporting.
        if is_distributed:
            t = torch.tensor([running_loss, float(n_steps)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            running_loss = float(t[0].item())
            n_steps = int(t[1].item())
        avg_loss = running_loss / max(1, n_steps)
        epoch_time = time.time() - t0

        # Val metrics (with debug) - evaluate on rank0 only to avoid duplicated work.
        val_metrics: Dict[str, float] = {}
        dbg: Dict[str, Any] = {}
        if is_main:
            eval_model = model.module if isinstance(model, DDP) else model
            eval_model.eval()
            with torch.no_grad():
                val_metrics, dbg = evaluate_stage2(
                    model=eval_model,
                    loader=val_loader,
                    device=device,
                    iou_thr=args.iou_pos_thr,
                    topk=ks,
                    return_debug=True,
                )
        if is_distributed:
            dist.barrier()

        if is_main:
            # print richer epoch summary
            msg = (
                f"[Epoch {epoch}/{args.epochs}] "
                f"loss={avg_loss:.4f} "
                f"top1_iou={val_metrics['top1_iou']:.3f} "
                f"max_iou_mean={val_metrics['max_iou_mean']:.3f} "
                f"ceiling@{args.iou_pos_thr:.2f}={val_metrics['stage1_ceiling_recall']:.3f} "
            )
            for k in ks:
                msg += f"recall@{k}={val_metrics[f'recall@{k}']:.3f} "
            print(msg.strip())

            # Save debug if any
            if dbg.bad_sample_ids:
                _save_json(
                    str(out_dir / "debug_bad_samples.json"),
                    {"bad_sample_ids": dbg.bad_sample_ids, "bad_reasons": dbg.bad_reasons},
                )
                print(f"[stage2][debug] wrote {len(dbg.bad_sample_ids)} bad sample ids to debug_bad_samples.json")

            # Save checkpoints (unwrap DDP)
            save_model = model.module if isinstance(model, DDP) else model
            ckpt = {
                "epoch": epoch,
                "model": save_model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "val": val_metrics,
            }
            torch.save(ckpt, out_dir / "ckpt_last.pt")

            key_val = float(val_metrics.get(best_key, 0.0))
            if key_val > best_val:
                best_val = key_val
                torch.save(ckpt, out_dir / "ckpt_best.pt")

            # Epoch-level wandb
            if wb is not None:
                wb.log(
                    {
                        "epoch": epoch,
                        "train/epoch_loss": float(avg_loss),
                        "train/epoch_time_sec": float(epoch_time),
                        **{f"val/{k}": float(v) for k, v in val_metrics.items()},
                    },
                    step=global_step,
                )

            _save_json(str(out_dir / "metrics_last.json"), {"epoch": epoch, "train_loss": avg_loss, **val_metrics})

            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"[stage2] reached max_steps={args.max_steps}, stopping after epoch {epoch}.")
                # ensure all ranks exit
                if is_distributed:
                    dist.barrier()
                break

        # keep ranks synchronized at epoch boundaries
        if is_distributed:
            dist.barrier()

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
