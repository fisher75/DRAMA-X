# DRAMA-X Fast System — Phase-1 (Swin-T + Single-Query)

## 你现在在做什么？
Phase-1 的目标只有一个：**把快系统端到端闭环跑通**。

> 真实 T 帧输入 → Video Swin-T backbone → Single-Query Joint Head → 输出 bbox(xyxy, 0~1) + risk(0~1)

做到这一步，你就能确定：
- 数据路径映射正确（URL → 本地帧目录）
- bbox 归一化正确（用真实图片尺寸）
- 模型/损失/优化器闭环正确（能 overfit 32 样本）

**只有 Phase-1 过了，Phase-2 才值得做（注入慢系统热力图/先验）。**

---

## 目录结构（已落地）
```
DRAMA-X/
  fast_system/
    drama_fast/
      dataset_phase1.py
      models/
        backbones.py
        head_query.py
        model_phase1.py
      train/
        train_overfit_32.py
      utils/
        path_resolver.py
```

---

## 依赖
建议最小依赖：
- torch
- torchvision (Video Swin-T 来自 torchvision.models.video.swin3d_t)
- pillow

安装示例：
```bash
pip install torch torchvision pillow
```

---

## 路径约定（你给的信息）
- fast sup jsonl：`annotation_coc/drama_x_fast_sup_v2_rule.jsonl`
- 数据根目录：`/data2/automan/data/drama_data`
  - 该目录下应存在 `combined/`
  - JSONL 里的 `image` URL 形如：`.../data/drama/combined/.../clip_xxx/frame_xxx.png`

`path_resolver.py` 会把 URL 的 `combined/...` 部分拼到你的数据根目录下。

---

## 运行：Overfit 32 Samples
在仓库根目录下：
```bash
cd fast_system
export DRAMA_DATA_ROOT=/data2/automan/data/drama_data

python -m drama_fast.train.train_overfit_32 \
  --jsonl ../annotation_coc/drama_x_fast_sup_v2_rule.jsonl \
  --data_root $DRAMA_DATA_ROOT \
  --num_frames 8 \
  --img_size 224 \
  --batch_size 4 \
  --epochs 50
```

成功标准：
- loss 明显下降，最后接近 0（尤其是 bbox loss）
- `pred_box` 与 `gt_box` 越来越接近

---

## 常见坑
1) **URL → 本地路径映射错**：会报找不到文件。先打印 `meta.keyframe_path` 检查是否真存在。
2) **bbox 归一化错**：一定要用 keyframe 的真实 `W,H`，不能硬编码 1920×1080。
3) **帧采样错**：如果 clip 目录里帧不全/命名不同，先 `ls` 看真实文件名模式。

---

## 下一步（Phase-2 预告）
Phase-2 会引入慢系统先验：
- 慢系统输出：关键对象热力图 / tubelet 提议 / 风险文本解释 → 转为数值 prior
- 快系统注入：对齐到 Swin 的 `(T',H',W')` 特征网格，做 additive / gated injection

Phase-1 先别碰这些，先把闭环跑通。
