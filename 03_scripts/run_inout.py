"""Design A — AIHub 실외/실내 서브셋 결합 학습 → D-Fire test.

통제: D-Fire ×3 + AIHub 14,122(클래스 SM:FL:NONE=1:1:1 매칭). inout만 변수.
셀: out → in → naive(혼합, 참고) 순. YOLO11s(D_full 11s=0.787과 비교), R8 동일 하이퍼파라미터.
가설: 실외 추가는 D_full(0.787~0.789) 이상, 실내는 무효/악화 → A-1 도메인 갭을 '실내 원인'으로 설명.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
IMGSZ, BATCH, DEVICE, EPOCHS = 640, 48, "0", 100
CELLS = [
    ("Combined_out_11s",   "combined_out.yaml"),
    ("Combined_in_11s",    "combined_in.yaml"),
    ("Combined_naive_11s", "combined_os.yaml"),
]


def metrics_dict(res) -> dict:
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None}


def run_cell(name, yaml):
    from ultralytics import YOLO
    data = str(BASE / "compositions" / yaml)
    best = Path(PROJECT) / name / "weights" / "best.pt"
    t0 = time.time()
    # 완료 판정: best.pt 존재 + optimizer 제거된(stripped) 크기(<40MB). 부분(미완, ~72MB)은 재학습.
    if not (best.exists() and best.stat().st_size < 40 * 1024 * 1024):
        YOLO("yolo11s.pt").train(
            data=data, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
            project=PROJECT, name=name, exist_ok=True, save_period=10,
            patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=3, cache=False, seed=0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=True)
    elapsed = time.time() - t0
    model = YOLO(str(best))
    tr = model.val(data=data, imgsz=IMGSZ, device=DEVICE, split="test",
                   project=PROJECT, name=f"{name}_test", exist_ok=True, plots=True, verbose=False)
    return {"name": name, "data": yaml, "elapsed_sec": round(elapsed, 1),
            "best_pt": str(best), "test": metrics_dict(tr)}


def main():
    out = Path(PROJECT) / "inout_summary.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    # 자가치유: best.pt 없거나 부분(>40MB, optimizer 포함)인 셀 기록 제거 → 재학습 대상
    def _complete(r):
        b = Path(r["best_pt"])
        return b.exists() and b.stat().st_size < 40 * 1024 * 1024
    results = [r for r in results if _complete(r)]
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    done = {r["name"] for r in results}
    for name, yaml in CELLS:
        if name in done:
            print(f"[{name}] 집계됨 — 건너뜀"); continue
        r = run_cell(name, yaml)
        results.append(r)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        t = r["test"]
        print(f"[{name}] DFire test mAP50={t['map50']:.4f} fireAP={t['fire_ap50']:.4f} smokeAP={t['smoke_ap50']:.4f}")
    print("\n===== Design A (DFire test, 참고 D_full 11s=0.787) =====")
    for r in results:
        print(f"{r['name']}: mAP50={r['test']['map50']:.4f}")


if __name__ == "__main__":
    main()
