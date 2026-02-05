import torch
import cv2
import numpy as np
import os
import sys

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fast_system.drama_fast.models.fast_system import FastSystemSwin
from fast_system.drama_fast.data.dataset_phase1 import DramaFastDataset

# CONFIG
JSONL = "/workspace/chz/code/DRAMA-X/annotation_coc/drama_x_fast_sup_v2_rule.jsonl"
DATA_ROOT = "/data2/automan/data/drama_data" # 你的真实路径
CHECKPOINT = "fast_system_overfit.pth" # 假设你刚才保存的模型叫这个
OUTPUT_DIR = "vis_results"
DEVICE = "cuda"

def draw_box(img, box, risk, color=(0, 0, 255)):
    # box: [x1, y1, x2, y2] normalized
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = int(x1*W), int(x2*W)
    y1, y2 = int(y1*H), int(y2*H)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"Risk: {risk:.2f}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model
    print("Loading Model...")
    model = FastSystemSwin().to(DEVICE)
    # 加载你刚才训练的权重 (如果刚才脚本没保存，你需要去 train_overfit 里加一行 torch.save)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT))
        print("Checkpoint loaded.")
    else:
        print("No checkpoint found! Running with random weights (Expect garbage).")

    model.eval()
    
    # 2. Load Data (Take first 5 samples)
    ds = DramaFastDataset(JSONL, DATA_ROOT, num_frames=8)
    
    print("Running Inference...")
    for i in range(5): # 看前5个
        sample = ds[i]
        img_tensor = sample['pixel_values'].unsqueeze(0).to(DEVICE) # [1, C, T, H, W]
        sample_id = sample['sample_id']
        
        # Inference
        with torch.no_grad():
            pred_box, pred_risk = model(img_tensor)
            
        # Unpack
        p_box = pred_box[0].cpu().numpy()
        p_risk = pred_risk[0].item()
        gt_box = sample['gt_boxes'].numpy()
        gt_risk = sample['gt_risk'].item()
        
        # Visualization (Take the last frame to draw)
        # img_tensor is [1, 3, 8, 224, 224]
        # We want last frame: [3, 224, 224] -> [224, 224, 3]
        vis_img = img_tensor[0, :, -1, :, :].permute(1, 2, 0).cpu().numpy()
        vis_img = (vis_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # Draw GT (Green)
        vis_img = draw_box(vis_img, gt_box, gt_risk, color=(0, 255, 0))
        # Draw Pred (Red)
        vis_img = draw_box(vis_img, p_box, p_risk, color=(0, 0, 255))
        
        save_path = os.path.join(OUTPUT_DIR, f"{sample_id}.jpg")
        cv2.imwrite(save_path, vis_img)
        print(f"Saved {save_path} | GT Risk: {gt_risk:.2f} | Pred Risk: {p_risk:.2f}")

if __name__ == "__main__":
    main()