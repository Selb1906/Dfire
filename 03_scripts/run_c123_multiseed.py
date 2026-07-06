"""AIHub C1~C3 멀티시드(n=3) — 4셀 mean±std 통일. seed0=기존 재사용, seed1·2만 재학습.

설정 100% 동일(YOLO11n, AdamW lr0 0.001, batch64, patience30), seed만 상이(C4 멀티시드와 짝).
평가: 각 시드 체크포인트를 (a)공유 val 19,080, (b)held-out test 7,632 두 셋에.
resumable(last.pt) + FOR_DISABLE(window-CLOSE 내성) + json 증분저장(중단 자가치유).
"""
from __future__ import annotations
import json, os, time, statistics as st
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
COMP = BASE / "compositions"
HELDOUT = str(COMP / "heldout_full.yaml")
IMGSZ, BATCH, DEVICE, EPOCHS = 640, 64, "0", 100
# (셀, 기존 seed0 run 이름, 학습 data.yaml)
CELLS = [
    ("C1", "AIHub_C1", r"D:\AIHub_Fire\yolo_071751_c1\data.yaml"),
    ("C2", "AIHub_C2", r"D:\AIHub_Fire\yolo_071751_c2\data.yaml"),
    ("C3", "AIHub_C3", str(COMP / "aihub_c3.yaml")),
]
SEEDS = [0, 1, 2]

HP = dict(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, project=PROJECT,
          exist_ok=True, save_period=10, patience=30, optimizer="AdamW",
          lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3, cache=False,
          hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
          mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=False)


def done(name):  # 11n stripped best ~5MB
    b = Path(PROJECT) / name / "weights" / "best.pt"
    return b.exists() and b.stat().st_size < 10 * 1024 * 1024


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]
    return {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "smoke_ap50": round(ap50[0], 4) if ap50 else None,   # AIHub 0=smoke
            "fire_ap50": round(ap50[1], 4) if len(ap50) > 1 else None}


def main():
    from ultralytics import YOLO
    out = Path(PROJECT) / "c123_multiseed.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for cell, base0, data in CELLS:
        res.setdefault(cell, {})
        for seed in SEEDS:
            if str(seed) in res[cell] and {"val", "heldout"} <= set(res[cell][str(seed)]):
                print(f"[{cell} s{seed}] 집계됨 — 건너뜀", flush=True); continue
            name = base0 if seed == 0 else f"{base0}_s{seed}"
            best = Path(PROJECT) / name / "weights" / "best.pt"
            last = Path(PROJECT) / name / "weights" / "last.pt"
            if seed != 0 and not done(name):   # seed0=재사용(학습 안 함)
                if last.exists():
                    print(f"[{cell} s{seed}] resume ({name})", flush=True)
                    YOLO(str(last)).train(resume=True)
                else:
                    print(f"[{cell} s{seed}] 학습 시작 ({name})", flush=True)
                    YOLO("yolo11n.pt").train(data=data, name=name, seed=seed, **HP)
            model = YOLO(str(best))
            ev = {}
            ev["val"] = metrics(model.val(data=data, imgsz=IMGSZ, device=DEVICE, split="val",
                                          project=PROJECT, name=f"{name}_valeval", exist_ok=True, plots=False, verbose=False))
            ev["heldout"] = metrics(model.val(data=HELDOUT, imgsz=IMGSZ, device=DEVICE, split="val",
                                              project=PROJECT, name=f"{name}_heldout", exist_ok=True, plots=False, verbose=False))
            res[cell][str(seed)] = ev
            out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{cell} s{seed}] val={ev['val']['map50']} heldout={ev['heldout']['map50']}", flush=True)
    print("\n===== C1~C3 멀티시드 요약 (val mAP50 mean±std) =====")
    for cell, _, _ in CELLS:
        v = [res[cell][str(s)]["val"]["map50"] for s in SEEDS if str(s) in res[cell]]
        if len(v) == 3:
            print(f"  {cell}: {st.mean(v):.4f}±{st.stdev(v):.4f}  seeds={v}")


if __name__ == "__main__":
    main()
