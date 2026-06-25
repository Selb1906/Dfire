"""E08 — DFire C4 데이터로 YOLO11n @imgsz=416 학습 (해상도 비교, §5.4 D-Fire 정합).

R8-C4(@640)와 동일 데이터(c4_balanced_nm: train 11,056)·동일 하이퍼파라미터, imgsz만 416.
평가: 동일 DFire val/test(원본 고정) → R8-C4(640)와 같은 형식으로 비교.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DATA = str(BASE / "compositions" / "c4_balanced_nm.yaml")
NAME = "E08_C4_416"
IMGSZ, BATCH, DEVICE, EPOCHS = 416, 64, "0", 100


def metrics_dict(res) -> dict:
    ap50 = [float(v) for v in res.box.ap50]
    r = [float(v) for v in res.box.r]
    return {
        "map50": float(res.box.map50), "map50_95": float(res.box.map),
        "precision": float(res.box.mp), "recall": float(res.box.mr),
        "smoke_ap50": ap50[0] if len(ap50) > 0 else None,
        "fire_ap50": ap50[1] if len(ap50) > 1 else None,
        "smoke_recall": r[0] if len(r) > 0 else None,
        "fire_recall": r[1] if len(r) > 1 else None,
    }


def main():
    from ultralytics import YOLO
    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    t0 = time.time()
    if best.exists():
        print(f"[{NAME}] best.pt 존재 → 평가만")
    else:
        YOLO("yolo11n.pt").train(
            data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
            project=PROJECT, name=NAME, exist_ok=True, save_period=10,
            patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, cache=False, seed=0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=True, plots=True)
    elapsed = time.time() - t0
    model = YOLO(str(best))
    val_res = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="val",
                        project=PROJECT, name=f"{NAME}_val", exist_ok=True, plots=True, verbose=False)
    test_res = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="test",
                         project=PROJECT, name=f"{NAME}_test", exist_ok=True, plots=True, verbose=False)
    out = {"name": NAME, "model": "yolo11n.pt", "imgsz": IMGSZ, "data": "c4_balanced_nm.yaml",
           "train_imgs": 11056, "elapsed_sec": round(elapsed, 1), "best_pt": str(best),
           "val": metrics_dict(val_res), "test": metrics_dict(test_res)}
    (Path(PROJECT) / "e08_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    t = out["test"]
    print(f"\n[{NAME}] 완료 — test mAP50={t['map50']:.4f} mAP50-95={t['map50_95']:.4f} "
          f"fireAP={t['fire_ap50']:.4f} smokeAP={t['smoke_ap50']:.4f} ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
