'''
设置数据根目录（真实路径）
export DRAMA_DATA_ROOT=/data2/automan/data/drama_data
先跑 overfit 32（必须先过）
python -m drama_fast.train.train_overfit_32 \
  --jsonl ../annotation_coc/drama_x_fast_sup_v2_rule.jsonl \
  --data_root $DRAMA_DATA_ROOT \
  --num_frames 8 \
  --img_size 224 \
  --batch_size 4 \
  --epochs 80 \
  --lr 2e-4
'''
import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

from drama_fast.dataset_phase1 import DramaFastDataset
from drama_fast.models.model_phase1 import FastSystemPhase1


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="annotation_coc/drama_x_fast_sup_v2_rule.jsonl")
    parser.add_argument("--data_root", type=str, default=os.environ.get("DRAMA_DATA_ROOT", "/data2/automan/data/drama_data"))
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_full = DramaFastDataset(
        jsonl_path=args.jsonl,
        data_root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size,
        stride=args.stride,
        topk=args.topk,
        return_meta=True,
    )

    ds = Subset(ds_full, list(range(min(32, len(ds_full)))))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = FastSystemPhase1(pretrained=True, freeze_backbone=args.freeze_backbone).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)

    print("== Overfit Debug Start ==")
    print(f"device={device} | jsonl={args.jsonl} | data_root={args.data_root}")
    print(f"subset={len(ds)} | batch={args.batch_size} | frames={args.num_frames} stride={args.stride}")

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        total_box = 0.0
        total_risk = 0.0

        for batch in dl:
            x = batch["pixel_values"].to(device)
            gt_box = batch["gt_box"].to(device)
            gt_risk = batch["gt_risk"].to(device)

            pred_box, pred_risk = model(x)

            # bbox loss: L1 + MSE tends to be stable in early debugging
            loss_box = F.l1_loss(pred_box, gt_box) + F.mse_loss(pred_box, gt_box)
            # risk loss: MSE (risk already in [0,1])
            loss_risk = F.mse_loss(pred_risk, gt_risk)

            loss = loss_box + 2.0 * loss_risk

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            total_box += loss_box.item()
            total_risk += loss_risk.item()

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            n = len(dl)
            print(f"Epoch {epoch:03d} | loss={total/n:.4f} box={total_box/n:.4f} risk={total_risk/n:.4f}")

            # print one sample for sanity
            try:
                b0 = next(iter(dl))
                with torch.no_grad():
                    pb, pr = model(b0["pixel_values"].to(device))
                print("  sample_id:", b0["meta"]["sample_id"][0])
                print("  gt_box   :", b0["gt_box"][0].tolist())
                print("  pred_box :", pb[0].detach().cpu().tolist())
                print("  gt_risk  :", float(b0["gt_risk"][0]))
                print("  pred_risk:", float(pr[0].detach().cpu()))
            except Exception as e:
                print("  (skip sample print)", e)

    print("== Overfit Debug Done ==")


if __name__ == "__main__":
    main()
