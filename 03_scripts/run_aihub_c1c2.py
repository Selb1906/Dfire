"""AIHub 071751 C1(fire-only)·C2(불균형 14:1) 학습 — §6.3 데이터셋 의존성 근거.

C4(균형+NM)와 동일 하이퍼파라미터·동일 val(공유, 자연분포) → C1/C2/C4 일관 비교.
목적: AIHub에서 C1 vs C2 순서 확인(원고 §6.3: 기존 R2/R3은 C2<C1, 테스트셋 불일치 주의).
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DEVICE, EPOCHS, IMGSZ, BATCH = "0", 100, 640, 64

CELLS = [
    ("AIHub_C1", r"D:\AIHub_Fire\yolo_071751_c1\data.yaml"),
    ("AIHub_C2", r"D:\AIHub_Fire\yolo_071751_c2\data.yaml"),
]


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


def run_cell(name, data):
    from ultralytics import YOLO
    best_pt = Path(PROJECT) / name / "weights" / "best.pt"
    t0 = time.time()
    if best_pt.exists():
        print(f"[{name}] best.pt 존재 → 학습 건너뜀, 평가만")
    else:
        m = YOLO("yolo11n.pt")
        print(f"[{name}] 학습 시작 — data={data}")
        m.train(data=data, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                project=PROJECT, name=name, exist_ok=True, save_period=10,
                patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
                warmup_epochs=3, cache=False, seed=0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
                mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=True, plots=True)
    elapsed = time.time() - t0
    model = YOLO(str(best_pt))
    val_res = model.val(data=data, imgsz=IMGSZ, device=DEVICE, split="val",
                        project=PROJECT, name=f"{name}_val", exist_ok=True,
                        plots=True, verbose=False)
    return {"name": name, "data": data, "elapsed_sec": round(elapsed, 1),
            "best_pt": str(best_pt), "val": metrics_dict(val_res)}


def main():
    out = Path(PROJECT) / "aihub_c1c2_summary.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["name"] for r in results}
    for name, data in CELLS:
        if name in done:
            print(f"[{name}] 이미 집계됨 — 건너뜀"); continue
        r = run_cell(name, data)
        results.append(r)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        v = r["val"]
        print(f"[{name}] val mAP50={v['map50']:.4f} smokeAP={v['smoke_ap50']:.4f} "
              f"fireAP={v['fire_ap50']:.4f} ({r['elapsed_sec']/3600:.2f}h)")
    print("\n===== AIHub C1 vs C2 (val) =====")
    for r in results:
        v = r["val"]
        print(f"{r['name']}: mAP50={v['map50']:.4f} smokeAP={v['smoke_ap50']:.4f} fireAP={v['fire_ap50']:.4f}")


if __name__ == "__main__":
    main()
