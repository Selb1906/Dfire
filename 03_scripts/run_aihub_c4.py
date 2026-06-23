"""AIHub 071751 C4(균형+NM) 1셀 학습 — DFire R8 보강(대규모 일반화 근거).

하이퍼파라미터는 DFire 4셀(run_4cells.py)과 동일하게 맞춰 비교 가능성 유지.
데이터: D:/AIHub_Fire/yolo_071751 (train 114K FL:SM 1:1+NM, val 19K 자연분포). 0=smoke,1=fire.
평가: val 만 (AIHub 071751은 Training/Validation 2분할 → 별도 test 없음).

결과:
  runs/AIHub_C4/weights/best.pt, confusion_matrix.png, PR_curve.png, results.csv
  runs/AIHub_C4_val/  (평가 그림)
  runs/aihub_c4_summary.json
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DATA = r"D:\AIHub_Fire\yolo_071751\data.yaml"
NAME = "AIHub_C4"
MODEL = "yolo11n.pt"
EPOCHS, IMGSZ, BATCH, DEVICE = 100, 640, 64, "0"


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
    Path(PROJECT).mkdir(parents=True, exist_ok=True)
    best_pt = Path(PROJECT) / NAME / "weights" / "best.pt"
    t0 = time.time()

    if best_pt.exists():
        print(f"[{NAME}] best.pt 존재 → 학습 건너뜀, 평가만")
    else:
        model = YOLO(MODEL)
        print(f"[{NAME}] 학습 시작 — data={DATA}, batch={BATCH}")
        model.train(
            data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
            project=PROJECT, name=NAME, exist_ok=True, save_period=10,
            patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, cache=False, seed=0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=True, plots=True,
        )

    elapsed = time.time() - t0
    model = YOLO(str(best_pt))
    val_res = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="val",
                        project=PROJECT, name=f"{NAME}_val", exist_ok=True,
                        plots=True, verbose=False)
    result = {"name": NAME, "model": MODEL, "data": DATA, "batch": BATCH,
              "train_imgs": 114462, "val_imgs": 19080,
              "elapsed_sec": round(elapsed, 1), "best_pt": str(best_pt),
              "val": metrics_dict(val_res)}
    out = Path(PROJECT) / "aihub_c4_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    v = result["val"]
    print(f"\n[{NAME}] 완료 — val mAP50={v['map50']:.4f} "
          f"smoke_AP={v['smoke_ap50']:.4f} fire_AP={v['fire_ap50']:.4f} "
          f"({elapsed/3600:.2f}h)\n집계: {out}")


if __name__ == "__main__":
    main()
