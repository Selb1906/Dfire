"""논문용 4셀 재학습 오케스트레이터 (DFire).

각 셀: 학습 → val 평가 → test 평가. best.pt·혼동행렬·PR곡선·results.csv 자동 보존.
하이퍼파라미터는 train.py(E-series) 기준. 클래스 0=smoke, 1=fire.

결과:
  runs/<name>/weights/best.pt          (가중치 — 삭제 금지)
  runs/<name>/confusion_matrix.png, PR_curve.png, results.csv  (학습 산출물)
  runs/<name>_val/, runs/<name>_test/  (평가 산출물: 혼동행렬·PR곡선)
  runs/4cell_summary.json              (모든 수치 집계)
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
CFG = BASE / "compositions"
PROJECT = str(BASE / "runs")

# (name, model, data.yaml, batch)
CELLS = [
    ("E_C1",     "yolo11n.pt", str(CFG / "c1_fire_only.yaml"),    64),
    ("E_C3",     "yolo11n.pt", str(CFG / "c3_balanced.yaml"),     64),
    ("E_C4",     "yolo11n.pt", str(CFG / "c4_balanced_nm.yaml"),  64),
    ("E_C4_11s", "yolo11s.pt", str(CFG / "c4_balanced_nm.yaml"),  48),
]

EPOCHS = 100
IMGSZ = 640
DEVICE = "0"


def metrics_dict(res) -> dict:
    """ultralytics val 결과 → 표준 딕셔너리. ap50 순서 = [smoke, fire]."""
    ap50 = [float(v) for v in res.box.ap50]
    r = [float(v) for v in res.box.r]
    return {
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "smoke_ap50": ap50[0] if len(ap50) > 0 else None,
        "fire_ap50": ap50[1] if len(ap50) > 1 else None,
        "smoke_recall": r[0] if len(r) > 0 else None,
        "fire_recall": r[1] if len(r) > 1 else None,
    }


def run_cell(name, model_pt, data, batch) -> dict:
    from ultralytics import YOLO

    best_pt = Path(PROJECT) / name / "weights" / "best.pt"
    t0 = time.time()

    if best_pt.exists():
        print(f"[{name}] best.pt 존재 → 학습 건너뜀, 평가만 수행")
    else:
        model = YOLO(model_pt)
        print(f"[{name}] 학습 시작 — model={model_pt}, data={Path(data).name}, batch={batch}")
        model.train(
            data=data, epochs=EPOCHS, imgsz=IMGSZ, batch=batch, device=DEVICE,
            project=PROJECT, name=name, exist_ok=True, save_period=10,
            patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, cache=False, seed=0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=True, plots=True,
        )

    elapsed = time.time() - t0
    # 평가 (best.pt 로 val + test)
    model = YOLO(str(best_pt))
    val_res = model.val(data=data, imgsz=IMGSZ, device=DEVICE, split="val",
                        project=PROJECT, name=f"{name}_val", exist_ok=True,
                        plots=True, verbose=False)
    test_res = model.val(data=data, imgsz=IMGSZ, device=DEVICE, split="test",
                         project=PROJECT, name=f"{name}_test", exist_ok=True,
                         plots=True, verbose=False)

    result = {
        "name": name, "model": model_pt, "data": Path(data).name,
        "batch": batch, "elapsed_sec": round(elapsed, 1),
        "best_pt": str(best_pt),
        "val": metrics_dict(val_res),
        "test": metrics_dict(test_res),
    }
    print(f"[{name}] 완료 — val mAP50={result['val']['map50']:.4f}  "
          f"test mAP50={result['test']['map50']:.4f}  ({elapsed/3600:.2f}h)")
    return result


def main():
    Path(PROJECT).mkdir(parents=True, exist_ok=True)
    summary_path = Path(PROJECT) / "4cell_summary.json"
    results = []
    if summary_path.exists():
        try:
            results = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            results = []
    done = {r["name"] for r in results}

    for name, model_pt, data, batch in CELLS:
        if name in done:
            print(f"[{name}] 이미 집계됨 — 건너뜀")
            continue
        res = run_cell(name, model_pt, data, batch)
        results.append(res)
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n===== 4셀 결과 요약 (test 기준) =====")
    print(f"{'cell':<10} {'model':<10} {'test_mAP50':>10} {'smoke_AP':>9} {'fire_AP':>8} {'P':>6} {'R':>6}")
    for r in results:
        t = r["test"]
        print(f"{r['name']:<10} {r['model']:<10} {t['map50']:>10.4f} "
              f"{(t['smoke_ap50'] or 0):>9.4f} {(t['fire_ap50'] or 0):>8.4f} "
              f"{t['precision']:>6.3f} {t['recall']:>6.3f}")
    print(f"\n집계 저장: {summary_path}")


if __name__ == "__main__":
    main()
