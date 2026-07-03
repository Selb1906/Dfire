"""AIHub C3(균형, NM 없음) 1셀 학습 — C1<C2<C3<C4 4셀 완성 + NM 효과(C3→C4) 분해.

C3 = C4에서 NM(빈 라벨) 제외 = 85,767장(FL포함=SM포함=57,391, 균형 유지). 별도 폴더 없이
C4의 images/train 중 라벨이 비어있지 않은 것만 리스트업(라벨은 C4 것 재사용).
설정·평가셋은 C1/C2/C4와 100% 동일(YOLO11n, seed0, val 19,080 공유). 목적: NM 단독 효과 격리.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
TR = Path(r"D:\AIHub_Fire\yolo_071751")
COMP = BASE / "compositions"
NAME = "AIHub_C3"
EPOCHS, IMGSZ, BATCH, DEVICE = 100, 640, 64, "0"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def build_c3_list():
    lst = COMP / "aihub_c3_train.txt"
    if lst.exists() and lst.stat().st_size > 0:
        n = sum(1 for _ in open(lst, encoding="utf-8"))
        print(f"[C3] 리스트 존재 재사용: {n}장"); return lst, n
    imgs, lbl = TR / "images" / "train", TR / "labels" / "train"
    keep = []
    for p in sorted(imgs.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lp = lbl / (p.stem + ".txt")
        if lp.exists() and lp.stat().st_size > 0:   # 비어있지 않은 라벨 = fire/smoke 포함(NM 제외)
            keep.append(str(p))
    lst.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"[C3] NM 제외 {len(keep)}장 리스트 생성 (기대 85,767)")
    return lst, len(keep)


def yaml_for(lst):
    y = COMP / "aihub_c3.yaml"
    y.write_text(
        "# AIHub C3(균형, NM 없음) — C4에서 NM 제외. val=공유 19,080. 0=smoke,1=fire.\n"
        f"train: {lst.as_posix()}\n"
        "val: D:/AIHub_Fire/yolo_071751/images/val\n\nnames: ['smoke', 'fire']\nnc: 2\n",
        encoding="utf-8")
    return str(y)


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "smoke_ap50": round(ap50[0], 4) if ap50 else None,
            "fire_ap50": round(ap50[1], 4) if len(ap50) > 1 else None}


def done():  # 11n stripped ~5MB
    b = Path(PROJECT) / NAME / "weights" / "best.pt"
    return b.exists() and b.stat().st_size < 10 * 1024 * 1024


def main():
    from ultralytics import YOLO
    lst, n = build_c3_list()
    data = yaml_for(lst)
    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    last = Path(PROJECT) / NAME / "weights" / "last.pt"
    t0 = time.time()
    if not done():
        if last.exists():   # 중단분 이어서(resume) — 저장된 args로 마지막 epoch부터 계속
            print(f"[{NAME}] 이어서 학습(resume) — {last}")
            YOLO(str(last)).train(resume=True)
        else:
            print(f"[{NAME}] 학습 시작 — {n}장, batch={BATCH}")
            YOLO("yolo11n.pt").train(
                data=data, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                project=PROJECT, name=NAME, exist_ok=True, save_period=10,
                patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
                warmup_epochs=3, cache=False, seed=0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
                mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=True)
    elapsed = round(time.time() - t0, 1)
    vr = YOLO(str(best)).val(data=data, imgsz=IMGSZ, device=DEVICE, split="val",
                             project=PROJECT, name=f"{NAME}_val", exist_ok=True, plots=True, verbose=False)
    out = {"name": NAME, "train_imgs": n, "val_imgs": 19080, "elapsed_sec": elapsed,
           "best_pt": str(best), "val": metrics(vr)}
    (Path(PROJECT) / "aihub_c3_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    v = out["val"]
    print(f"\n[{NAME}] val mAP50={v['map50']} smokeAP={v['smoke_ap50']} fireAP={v['fire_ap50']} ({elapsed/3600:.2f}h)")
    print("  비교: C1 0.463 / C2 0.770 / C4 0.913 (공유 val 19,080)")


if __name__ == "__main__":
    main()
