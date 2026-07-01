"""결합 학습(오버샘플) — D-Fire ×3 지배 + AIHub 보충 → D-Fire test 평가.

목적: 도메인 갭 고려해 D-Fire를 지배적으로 한 결합이 D_full(0.789)/AIHub단독(0.262)보다 나은지.
모델 YOLO11m(D_full best와 비교). R8 동일 하이퍼파라미터. 평가 D-Fire val/test.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DATA = str(BASE / "compositions" / "combined_os.yaml")
NAME = "Combined_os_11m"
IMGSZ, BATCH, DEVICE, EPOCHS = 640, 32, "0", 100


def metrics_dict(res) -> dict:
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None}


def main():
    from ultralytics import YOLO
    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    t0 = time.time()
    if not best.exists():
        YOLO("yolo11m.pt").train(
            data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
            project=PROJECT, name=NAME, exist_ok=True, save_period=10,
            patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, cache=False, seed=0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=True, plots=True)
    elapsed = time.time() - t0
    model = YOLO(str(best))
    vr = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="val",
                   project=PROJECT, name=f"{NAME}_val", exist_ok=True, plots=True, verbose=False)
    tr = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="test",
                   project=PROJECT, name=f"{NAME}_test", exist_ok=True, plots=True, verbose=False)
    out = {"name": NAME, "data": "D-Fire x3 + AIHub sample (56,488, D-Fire 75%)",
           "elapsed_sec": round(elapsed, 1), "best_pt": str(best),
           "val": metrics_dict(vr), "test": metrics_dict(tr)}
    (Path(PROJECT) / "combined_os_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    t = out["test"]
    print(f"\n[{NAME}] DFire test mAP50={t['map50']:.4f} fireAP={t['fire_ap50']:.4f} smokeAP={t['smoke_ap50']:.4f} ({elapsed/3600:.2f}h)")
    print(f"  비교: D_full 11m=0.789 / AIHub단독→DFire test=0.262")


if __name__ == "__main__":
    main()
