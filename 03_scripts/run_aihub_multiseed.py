"""AIHub C4 멀티시드(n=3) — seed 0(기존 AIHub_C4) + seed 1·2 재학습 → 5개 val셋 평가.

목적: 헤드라인 수치(전체 0.913 / 실내 0.946·0.958 / 실외 0.824·0.821)의 통계적 확정(mean±std).
통제: C4와 100% 동일 설정(YOLO11n, data.yaml 114,462, batch64, ep100/patience30, AdamW lr0.001), seed만 0→1→2.
평가셋(재학습 없이 재사용): 전체 val / 실내(raw) / 실외(raw) / 실내(1:1:1 균형) / 실외(1:1:1 균형).
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DATA = r"D:\AIHub_Fire\yolo_071751\data.yaml"
COMP = BASE / "compositions"
IMGSZ, BATCH, DEVICE, EPOCHS = 640, 64, "0", 100
SEEDS = [(0, "AIHub_C4"), (1, "AIHub_C4_s1"), (2, "AIHub_C4_s2")]
EVALSETS = [("full", "aihub_val.yaml"), ("in", "aihub_val_in.yaml"), ("out", "aihub_val_out.yaml"),
            ("in_bal", "aihub_val_in_bal.yaml"), ("out_bal", "aihub_val_out_bal.yaml")]

HP = dict(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, project=PROJECT,
          exist_ok=True, save_period=10, patience=30, optimizer="AdamW",
          lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3, cache=False,
          hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
          mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=False)


def done(name):  # 11n stripped best.pt ~5MB, 부분(optimizer 포함) ~15MB → <10MB=완료
    b = Path(PROJECT) / name / "weights" / "best.pt"
    return b.exists() and b.stat().st_size < 10 * 1024 * 1024


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]
    return {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "smoke_ap50": round(ap50[0], 4) if ap50 else None,
            "fire_ap50": round(ap50[1], 4) if len(ap50) > 1 else None}


def main():
    from ultralytics import YOLO
    out = Path(PROJECT) / "aihub_multiseed.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for seed, name in SEEDS:
        best = Path(PROJECT) / name / "weights" / "best.pt"
        t0 = time.time()
        if not done(name):
            print(f"[seed {seed}] 학습 시작 ({name})", flush=True)
            YOLO("yolo11n.pt").train(data=DATA, name=name, seed=seed, **HP)
        elapsed = round(time.time() - t0, 1)
        model = YOLO(str(best))
        ev = {}
        for tag, yml in EVALSETS:
            r = model.val(data=str(COMP / yml), imgsz=IMGSZ, device=DEVICE, split="val",
                          project=PROJECT, name=f"{name}_{tag}", exist_ok=True, plots=False, verbose=False)
            ev[tag] = metrics(r)
            print(f"  [seed {seed}] {tag}: mAP50={ev[tag]['map50']}", flush=True)
        results[str(seed)] = {"name": name, "elapsed_sec": elapsed, "eval": ev}
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== 멀티시드 완료 (seed별 전체 val mAP50) =====")
    for s in sorted(results):
        print(f"  seed {s}: full={results[s]['eval']['full']['map50']} "
              f"in_bal={results[s]['eval']['in_bal']['map50']} out_bal={results[s]['eval']['out_bal']['map50']}")


if __name__ == "__main__":
    main()
