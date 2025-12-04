from datasets import load_dataset
import os

out_dir = "drama_intent"
os.makedirs(out_dir, exist_ok=True)

ds = load_dataset("mgod96/DRAMA-X", split="train")  # 官方数据集
out_path = os.path.join(out_dir, "updated_output.json")
ds.to_json(out_path)
print("Saved to:", out_path)
