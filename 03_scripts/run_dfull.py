"""D_full — 최종/배포 모델 (전체 DFire train 14,122, 다운샘플 없음) → DFire test.

★ ablation C1~C4와 완전 별개 실험. C4는 균형 위해 fire를 3,828로 다운샘플한 통제 셀,
  D_full은 전체 데이터 + 큰 모델로 절대 성능(헤드라인/Table 6)을 노리는 별개 최종 모델.
YOLO11s/11m/11l 3종 학습 → best 선정. R8 동일 하이퍼파라미터(seed 0).
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DATA = str(BASE / "compositions" / "dfull.yaml")
IMGSZ, DEVICE, EPOCHS = 640, "0", 100
# (run 이름, 모델, batch) — 큰 모델일수록 batch 축소
MODELS = [
    ("Dfull_11s", "yolo11s.pt", 48),
    ("Dfull_11m", "yolo11m.pt", 32),
    ("Dfull_11l", "yolo11l.pt", 16),
]


def metrics_dict(res) -> dict:
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None,
            "smoke_recall": r[0] if r else None,
            "fire_recall": r[1] if len(r) > 1 else None}


def run_one(name, model_pt, batch):
    from ultralytics import YOLO
    best = Path(PROJECT) / name / "weights" / "best.pt"
    t0 = time.time()
    if not best.exists():
        try:
            YOLO(model_pt).train(
                data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=batch, device=DEVICE,
                project=PROJECT, name=name, exist_ok=True, save_period=10,
                patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
                warmup_epochs=3, cache=False, seed=0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
                mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=True)
        except Exception as e:
            print(f"[{name}] 학습 실패(발산 등): {e}")
            return {"name": name, "model": model_pt, "batch": batch, "failed": str(e)}
    if not best.exists():
        return {"name": name, "model": model_pt, "batch": batch, "failed": "no best.pt"}
    model = YOLO(str(best))
    vr = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="val",
                   project=PROJECT, name=f"{name}_val", exist_ok=True, plots=True, verbose=False)
    tr = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, split="test",
                   project=PROJECT, name=f"{name}_test", exist_ok=True, plots=True, verbose=False)
    return {"name": name, "model": model_pt, "batch": batch,
            "elapsed_sec": round(time.time() - t0, 1), "best_pt": str(best),
            "val": metrics_dict(vr), "test": metrics_dict(tr)}


def main():
    out = Path(PROJECT) / "dfull_summary.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["name"] for r in results}
    for name, model_pt, batch in MODELS:
        if name in done:
            print(f"[{name}] 집계됨 — 건너뜀"); continue
        r = run_one(name, model_pt, batch)
        results.append(r)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        if "test" in r:
            print(f"[{name}] DFire test mAP50={r['test']['map50']:.4f} "
                  f"fireAP={r['test']['fire_ap50']:.4f} smokeAP={r['test']['smoke_ap50']:.4f}")
    print("\n===== D_full (DFire test, 전체 데이터) =====")
    best = None
    for r in results:
        if "test" not in r:
            print(f"{r['name']}: 실패"); continue
        m = r["test"]["map50"]
        print(f"{r['name']}: mAP50={m:.4f}")
        if best is None or m > best[1]:
            best = (r["name"], m)
    if best:
        print(f"\n>>> BEST: {best[0]} = {best[1]:.4f} (vs C4 ablation 0.740)")


if __name__ == "__main__":
    main()
