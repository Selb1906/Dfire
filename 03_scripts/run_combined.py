"""요청 A-2 — AIHub 071751 train + DFire full train 결합 학습 → DFire test 평가.

결합: AIHub 114,462 + DFire 14,122 = 128,584 (단순 병합, ultralytics 다중 train 소스).
평가: DFire val(3,099) / test(4,306). 클래스 0=smoke,1=fire. R8 동일 하이퍼파라미터.
목적: 대규모+도메인 데이터로 DFire test 경쟁 수치 확보 시도.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DATA = str(BASE / "compositions" / "combined_aihub_dfire.yaml")
NAME = "E21_combined"
IMGSZ, BATCH, DEVICE, EPOCHS = 640, 64, "0", 100


def metrics_dict(res) -> dict:
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None,
            "smoke_recall": r[0] if r else None,
            "fire_recall": r[1] if len(r) > 1 else None}


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
    vr = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="val",
                   project=PROJECT, name=f"{NAME}_val", exist_ok=True, plots=True, verbose=False)
    tr = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="test",
                   project=PROJECT, name=f"{NAME}_test", exist_ok=True, plots=True, verbose=False)
    out = {"name": NAME, "data": "combined AIHub114K+DFire14K", "train_imgs": 128584,
           "elapsed_sec": round(elapsed, 1), "best_pt": str(best),
           "val": metrics_dict(vr), "test": metrics_dict(tr)}
    (Path(PROJECT) / "combined_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    t = out["test"]
    print(f"\n[{NAME}] 완료 — DFire test mAP50={t['map50']:.4f} "
          f"fireAP={t['fire_ap50']:.4f} smokeAP={t['smoke_ap50']:.4f} ({elapsed/3600:.2f}h)")
    print(f"  (참고) DFire 단독 C4=0.740 / AIHub 단독→DFire test=0.262")


if __name__ == "__main__":
    main()
