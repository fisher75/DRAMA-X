import torch
from torch.utils.data import DataLoader
from drama_fast.dataset_phase1 import DramaFastDataset
from drama_fast.models.model_phase1 import FastSystemPhase1

# 配置 (与你训练时保持一致)
JSONL = "./splits_v3/train.jsonl"
ROOT = "/data2/automan/data/drama_data"
IMG_SIZE = 224
NUM_QUERIES = 8  # 你训练设定的 query 数量
TOPK = 5         # 你训练设定的 topk

def run_diagnosis():
    print(f"\n=== 开始诊断 (Query={NUM_QUERIES}, TopK={TOPK}) ===")
    
    # 1. 加载数据
    print("Loading Dataset...")
    ds = DramaFastDataset(JSONL, ROOT, num_frames=8, img_size=IMG_SIZE, topk_targets=TOPK)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    
    # 2. 打印 GT 信息
    gt_boxes = batch['gt_boxes_topk'] # [B, K, 4]
    gt_mask = batch['gt_mask_topk']   # [B, K]
    
    print(f"\n[Data Shapes]")
    print(f"gt_boxes_topk.shape: {gt_boxes.shape} (期望: [4, {TOPK}, 4])")
    print(f"有效目标数量 (per sample): {gt_mask.sum(dim=1).tolist()}")
    
    # 3. 加载模型 & 前向传播
    print("\n[Model Forward]")
    model = FastSystemPhase1(img_size=IMG_SIZE, num_queries=NUM_QUERIES, pretrained=False)
    model.eval()
    
    with torch.no_grad():
        out = model(batch['pixel_values'])
        # 处理可能的返回值 (单返回值 or Tuple)
        if isinstance(out, tuple):
            pred_boxes, pred_risks = out
        else:
            print("❌ 模型似乎返回了单值 (Single Query?)，请检查 num_queries 设置")
            return

    print(f"pred_boxes.shape:    {pred_boxes.shape} (期望: [4, {NUM_QUERIES}, 4])")
    print(f"pred_risks.shape:    {pred_risks.shape} (期望: [4, {NUM_QUERIES}])")
    
    # 4. 模拟匹配逻辑 (诊断是否维度错乱)
    # 这是一个简化的检查，看是否存在维度广播错误
    try:
        print("\n[逻辑预演]")
        # 假设简单的贪心匹配或 Cost 计算
        cost_matrix_shape = (pred_boxes.shape[1], gt_boxes.shape[1]) # Q x K
        print(f"Cost Matrix 理论形状 (Q x K): {cost_matrix_shape}")
        
        # 检查是否所有 Query 都参与了 Loss 计算
        print("如果 GPT 问 'box loss 参与数量'：")
        print(f"  - 若代码写错，可能是 B*Q = {4*NUM_QUERIES} (错误! Unmatched 也算了)")
        print(f"  - 若代码正确，应该是 B*Matched ≈ {int(gt_mask.sum().item())} (正确)")
        
    except Exception as e:
        print(f"诊断过程出错: {e}")

if __name__ == "__main__":
    run_diagnosis()