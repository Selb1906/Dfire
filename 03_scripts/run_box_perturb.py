"""B3 — 평가측 박스 섭동 민감도. C4 예측은 1회(고정), GT 박스만 ±10/20% 스케일 → CPU 재채점.

교수님 지적("박스 크기 판단이 사람마다 달라 AP가 달라진다") 정량 답.
GPU: C4 predict 1회(대표 실내 subset 4,000). 이후 5개 GT변형 mAP@0.5는 전부 CPU.
출력: runs/box_perturb.json + 콘솔.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
BASE = Path(r"C:\YangHyunHo\DFire")
C4 = str(BASE / "runs" / "AIHub_C4" / "weights" / "best.pt")
IN_LIST = BASE / "compositions" / "aihub_val_in_list.txt"
N_SUB = 4000
SCALES = [1.0, 0.9, 1.1, 0.8, 1.2]   # GT 박스 폭·높이 스케일
IOU_T = 0.5


def img_to_label(p):
    return Path(p.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")


def load_gt(lp):
    out = []
    try:
        for l in lp.read_text().splitlines():
            if not l.strip():
                continue
            c, cx, cy, w, h = int(l.split()[0]), *map(float, l.split()[1:5])
            out.append([c, cx, cy, w, h])
    except Exception:
        pass
    return out


def xywhn_to_xyxy(c, cx, cy, w, h):
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap_all_point(tp, conf, n_gt):
    """단일 클래스 AP@0.5 (all-point interpolation)."""
    if n_gt == 0:
        return None
    order = np.argsort(-np.asarray(conf))
    tp = np.asarray(tp)[order]
    fp = 1 - tp
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate(([0], rec, [1]))
    mpre = np.concatenate(([0], prec, [0]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))


def map50(preds, gts, scale):
    """preds: {img:[(cls,conf,xyxy)]}, gts:{img:[[c,cx,cy,w,h]]}. GT 박스 w·h ×scale."""
    classes = [0, 1]
    aps = []
    for cls in classes:
        tp, conf, n_gt = [], [], 0
        for img in preds:
            g = [xywhn_to_xyxy(c, cx, cy, min(w*scale, 1.0), min(h*scale, 1.0))
                 for c, cx, cy, w, h in gts.get(img, []) if c == cls]
            n_gt += len(g)
            matched = [False]*len(g)
            pr = sorted([p for p in preds[img] if p[0] == cls], key=lambda x: -x[1])
            for _, cf, box in pr:
                best, bi = IOU_T, -1
                for gi, gb in enumerate(g):
                    if matched[gi]:
                        continue
                    v = iou(box, gb)
                    if v >= best:
                        best, bi = v, gi
                if bi >= 0:
                    matched[bi] = True; tp.append(1)
                else:
                    tp.append(0)
                conf.append(cf)
        ap = ap_all_point(tp, conf, n_gt)
        if ap is not None:
            aps.append(ap)
    return round(float(np.mean(aps)), 4) if aps else 0.0


def main():
    from ultralytics import YOLO
    imgs = [l.strip() for l in IN_LIST.read_text(encoding="utf-8").splitlines() if l.strip()][:N_SUB]
    print(f"C4 예측(실내 subset {len(imgs)}) — GPU 1회", flush=True)
    model = YOLO(C4)
    preds, gts = {}, {}
    for i in range(0, len(imgs), 200):
        batch = imgs[i:i+200]
        res = model.predict(batch, imgsz=640, device="0", conf=0.001, verbose=False, stream=False)
        for ip, r in zip(batch, res):
            b = r.boxes
            preds[ip] = [(int(c), float(cf), [float(x) for x in xy])
                         for c, cf, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())]
            gts[ip] = load_gt(img_to_label(ip))
    print("예측 완료 → CPU 재채점", flush=True)
    table = {}
    base = map50(preds, gts, 1.0)
    for s in SCALES:
        m = map50(preds, gts, s)
        table[f"scale_{s:.1f}"] = {"map50": m, "delta_vs_base": round(m - base, 4)}
        print(f"  GT×{s:.1f}: mAP@0.5={m}  (Δ {m-base:+.4f})", flush=True)
    out = {"checkpoint": "AIHub_C4", "n_images": len(imgs), "iou_thr": IOU_T,
           "base_scale_1.0": base, "table": table}
    (BASE / "runs" / "box_perturb.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[저장] runs/box_perturb.json")


if __name__ == "__main__":
    main()
