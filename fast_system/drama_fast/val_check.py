import json, numpy as np
p="./runs/phase1_swin_t/eval_val.json"
rows=json.load(open(p,"r"))

def inv_ratio(key="pred_box"):
    bad=0
    for r in rows:
        x1,y1,x2,y2=r[key]
        if (x2<=x1) or (y2<=y1):
            bad+=1
    return bad, len(rows), bad/len(rows)

bad,n,ratio = inv_ratio("pred_box")
print("pred inverted:", bad, "/", n, "ratio=", ratio)

# 也看一下 box 的宽高分布是否离谱（接近0 或 接近1）
ws=[]; hs=[]
for r in rows:
    x1,y1,x2,y2=r["pred_box"]
    ws.append(x2-x1); hs.append(y2-y1)
ws=np.array(ws); hs=np.array(hs)
print("pred w: mean",ws.mean(),"p10",np.percentile(ws,10),"p90",np.percentile(ws,90))
print("pred h: mean",hs.mean(),"p10",np.percentile(hs,10),"p90",np.percentile(hs,90))