"""T1 — 전이학습(pretrain→finetune) vs 혼합(Combined) vs 타깃단독(D_full).

데이터 고정(D-Fire 14,122 + AIHub-mixed 14,122 = Combined_naive와 동일), 통합 '전략'만 변경:
  Stage1: yolo11s.pt → AIHub 14K 사전학습 (COCO init에서 소스 도메인 특징 학습)
  Stage2: Stage1 best → D-Fire 14,122 미세조정 (마지막 학습이 타깃 → 도메인 정렬)
평가: D-Fire test. 비교: Combined_naive(혼합, 동일 데이터) / D_full 11s(0.787, 타깃단독).
하이퍼파라미터는 R8/D_full과 동일 — 초기화(가중치)만 다르게 하여 '전략' 효과를 격리.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
AIHUB14K = str(BASE / "compositions" / "aihub14k.yaml")
DFULL = str(BASE / "compositions" / "dfull.yaml")
IMGSZ, BATCH, DEVICE, EPOCHS = 640, 48, "0", 100
PRETRAIN, FINETUNE = "Transfer_pretrain_aihub14k", "Transfer_finetune_dfire"

HP = dict(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, project=PROJECT,
          exist_ok=True, save_period=10, patience=30, optimizer="AdamW",
          lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3, cache=False, seed=0,
          hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
          mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=True)


def done(name):  # stripped(<40MB) best.pt = 완료
    b = Path(PROJECT) / name / "weights" / "best.pt"
    return b.exists() and b.stat().st_size < 40 * 1024 * 1024


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None}


def main():
    from ultralytics import YOLO
    t0 = time.time()
    # Stage1 — AIHub 14K 사전학습
    if not done(PRETRAIN):
        YOLO("yolo11s.pt").train(data=AIHUB14K, name=PRETRAIN, **HP)
    pre_best = str(Path(PROJECT) / PRETRAIN / "weights" / "best.pt")

    # Stage2 — D-Fire 미세조정 (사전학습 가중치에서 출발)
    if not done(FINETUNE):
        YOLO(pre_best).train(data=DFULL, name=FINETUNE, **HP)
    ft_best = str(Path(PROJECT) / FINETUNE / "weights" / "best.pt")

    tr = YOLO(ft_best).val(data=DFULL, imgsz=IMGSZ, device=DEVICE, split="test",
                           project=PROJECT, name=f"{FINETUNE}_test", exist_ok=True,
                           plots=True, verbose=False)
    out = {"name": "Transfer_11s", "strategy": "pretrain(AIHub14K)->finetune(DFire)",
           "elapsed_sec": round(time.time() - t0, 1), "best_pt": ft_best,
           "pretrain_best": pre_best, "test": metrics(tr)}
    (Path(PROJECT) / "transfer_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    t = out["test"]
    print(f"\n[Transfer_11s] DFire test mAP50={t['map50']:.4f} "
          f"fireAP={t['fire_ap50']:.4f} smokeAP={t['smoke_ap50']:.4f}")
    print("  비교: D_full 11s=0.787(타깃단독) / Combined_naive(혼합, inout_summary.json)")


if __name__ == "__main__":
    main()
