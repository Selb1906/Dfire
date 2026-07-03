"""IFireSmoke (Sozol et al., PLOS ONE) 네이티브 직접 벤치 — §5.9 '인용→직접비교' 강화.

우리 파이프라인(YOLO11n, R8/AIHub 동일 하이퍼파라미터)을 그들 train 4000에 학습 → 그들 test 500 평가.
클래스 원본순서 유지(0=Fire,1=Smoke). 목적: 같은 데이터 위에서 그들 Enhanced-YOLOv5 보고치와 head-to-head.
주의: 이 데이터셋은 이미 균형·NM 0장이라 우리 '구성방법(균형+NM)'이 아닌 '모델/파이프라인' 직접비교임.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")  # window-CLOSE forrtl abort 방지

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DATA = r"C:\YangHyunHo\DFire\ifiresmoke\IndoorFS\ifs.yaml"
NAME = "IFireSmoke_11n"
EPOCHS, IMGSZ, BATCH, DEVICE = 100, 640, 64, "0"


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "fire_ap50": round(ap50[0], 4) if ap50 else None,       # 0=Fire
            "smoke_ap50": round(ap50[1], 4) if len(ap50) > 1 else None}  # 1=Smoke


def done():
    b = Path(PROJECT) / NAME / "weights" / "best.pt"
    return b.exists() and b.stat().st_size < 10 * 1024 * 1024


def main():
    from ultralytics import YOLO
    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    last = Path(PROJECT) / NAME / "weights" / "last.pt"
    t0 = time.time()
    if not done():
        if last.exists():
            print(f"[{NAME}] 이어서 학습(resume)"); YOLO(str(last)).train(resume=True)
        else:
            print(f"[{NAME}] 학습 시작 — train 4000")
            YOLO("yolo11n.pt").train(
                data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                project=PROJECT, name=NAME, exist_ok=True, save_period=10,
                patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
                warmup_epochs=3, cache=False, seed=0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
                mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=True)
    elapsed = round(time.time() - t0, 1)
    model = YOLO(str(best))
    vr = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="val",
                   project=PROJECT, name=f"{NAME}_val", exist_ok=True, plots=True, verbose=False)
    tr = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="test",
                   project=PROJECT, name=f"{NAME}_test", exist_ok=True, plots=True, verbose=False)
    out = {"name": NAME, "dataset": "IFireSmoke (Sozol et al., HF shahriar-5/IFireSmoke)",
           "train_imgs": 4000, "test_imgs": 500, "classes": "0=Fire,1=Smoke",
           "elapsed_sec": elapsed, "best_pt": str(best),
           "val": metrics(vr), "test": metrics(tr)}
    (Path(PROJECT) / "ifiresmoke_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    t = out["test"]
    print(f"\n[{NAME}] test mAP50={t['map50']} fireAP={t['fire_ap50']} smokeAP={t['smoke_ap50']} ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
