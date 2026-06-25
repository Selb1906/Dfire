"""Option B — 데이터량 통제 실험 (리뷰어 ①번 방어): NM vs 신호, 동일 총량 7,664.

E09_C3_vol : 전체 신호(NM 0) 7,664
E10_C4_eq  : C3 + NM 3,066 = 7,664 (C3_vol과 동일 총량, 추가분만 NM)
C4_eq > C3_vol 이면 "NM이 동일 분량 신호보다 우월" → NM 효과를 총량과 분리 입증.
R8 셀과 동일 하이퍼파라미터·동일 DFire val/test(4,306) 평가.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DEVICE, EPOCHS, IMGSZ, BATCH = "0", 100, 640, 64
CELLS = [
    ("E09_C3_vol", str(BASE / "compositions" / "c3_vol.yaml")),
    ("E10_C4_eq",  str(BASE / "compositions" / "c4_eq.yaml")),
]


def metrics_dict(res) -> dict:
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None,
            "smoke_recall": r[0] if r else None,
            "fire_recall": r[1] if len(r) > 1 else None}


def run_cell(name, data):
    from ultralytics import YOLO
    best = Path(PROJECT) / name / "weights" / "best.pt"
    t0 = time.time()
    if best.exists():
        print(f"[{name}] best.pt 존재 → 평가만")
    else:
        YOLO("yolo11n.pt").train(
            data=data, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
            project=PROJECT, name=name, exist_ok=True, save_period=10,
            patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, cache=False, seed=0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=True, plots=True)
    elapsed = time.time() - t0
    model = YOLO(str(best))
    vr = model.val(data=data, imgsz=IMGSZ, device=DEVICE, split="val",
                   project=PROJECT, name=f"{name}_val", exist_ok=True, plots=True, verbose=False)
    tr = model.val(data=data, imgsz=IMGSZ, device=DEVICE, split="test",
                   project=PROJECT, name=f"{name}_test", exist_ok=True, plots=True, verbose=False)
    return {"name": name, "data": Path(data).name, "elapsed_sec": round(elapsed, 1),
            "best_pt": str(best), "val": metrics_dict(vr), "test": metrics_dict(tr)}


def main():
    out = Path(PROJECT) / "volcontrol_summary.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["name"] for r in results}
    for name, data in CELLS:
        if name in done:
            print(f"[{name}] 이미 집계됨 — 건너뜀"); continue
        r = run_cell(name, data); results.append(r)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        t = r["test"]
        print(f"[{name}] test mAP50={t['map50']:.4f} fireAP={t['fire_ap50']:.4f} smokeAP={t['smoke_ap50']:.4f}")
    print("\n===== Option B (test) =====")
    for r in results:
        t = r["test"]; print(f"{r['name']}: mAP50={t['map50']:.4f} fireAP={t['fire_ap50']:.4f} smokeAP={t['smoke_ap50']:.4f}")


if __name__ == "__main__":
    main()
