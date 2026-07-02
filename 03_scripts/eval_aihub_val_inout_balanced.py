"""요청A — 클래스 구성 통제(SM:FL:NONE=1:1:1) 실내/실외 val 재평가 (재학습 없음).

배경: 앞선 재평가에서 실내 0.946 > 실외 0.824였으나, 실내가 FL편중(fire AP 0.980)이라 '화염 편중 교란' 우려.
방법: Design A와 동일한 클래스 매칭(even_sample, SM:FL:NONE=1:1:1)을 val 실내/실외에 적용 → 두 서브셋의
      클래스 구성을 동일하게 만든 뒤 AIHub C4로 각각 val(). per = 6버킷(inout×class) 최소값으로 자동 결정.
목적: "실내>실외"가 클래스 편중 때문인지, 클래스 통제 후에도 남는 실질 도메인 차이인지 규명.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
VAL_IMG = Path(r"D:\AIHub_Fire\yolo_071751\images\val")
VAL_LABELS = Path(r"D:\AIHub_Fire\extracted\Validation_labels")
BRANCHES = ["화재 현상", "화재현상"]
C4 = str(BASE / "runs" / "AIHub_C4" / "weights" / "best.pt")
OUTDIR = BASE / "compositions"
IMG_EXT = {".jpg", ".jpeg", ".png"}
CLASSES = ("SM", "FL", "NONE")
IMGSZ, DEVICE = 640, "0"


def prefix(stem):
    return "_".join(stem.split("_")[:3])


def even_sample(items, k):   # Design A와 동일
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def build_inout_map():
    branch = next((VAL_LABELS / b for b in BRANCHES if (VAL_LABELS / b).exists()), None)
    inout, seen = {}, set()
    for jp in branch.rglob("JSON/*.json"):
        clip = jp.parent.parent.name
        if clip in seen:
            continue
        seen.add(clip)
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception:
            continue
        fn = d.get("image", {}).get("filename", jp.stem + ".jpg")
        inout[prefix(Path(fn).stem)] = (d.get("attributes", {}) or {}).get("inout")
    return inout


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]
    return {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "smoke_ap50": round(ap50[0], 4) if ap50 else None,
            "fire_ap50": round(ap50[1], 4) if len(ap50) > 1 else None}


def main():
    from ultralytics import YOLO
    inout = build_inout_map()
    bucket = defaultdict(list)   # (io, cls) → [경로]
    for p in sorted(VAL_IMG.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        io = inout.get(prefix(p.stem))
        cls = p.stem.split("_")[1] if len(p.stem.split("_")) > 1 else "?"
        if io in ("in", "out") and cls in CLASSES:
            bucket[(io, cls)].append(str(p))
    sizes = {(io, c): len(bucket[(io, c)]) for io in ("in", "out") for c in CLASSES}
    print("버킷 크기:", sizes)
    per = min(sizes.values())
    print(f"클래스당 매칭 표본 per = {per} → 각 서브셋 {per*len(CLASSES)}장 (SM:FL:NONE=1:1:1)")

    model = YOLO(C4)
    results = {"checkpoint": C4, "per_class": per, "n_each": per * len(CLASSES),
               "note": "클래스 통제(1:1:1) 실내/실외 val 재평가", "bucket_sizes": {f"{k[0]}_{k[1]}": v for k, v in sizes.items()}}
    for io in ("in", "out"):
        sub = []
        for c in CLASSES:
            sub += even_sample(bucket[(io, c)], per)
        lst = OUTDIR / f"aihub_val_{io}_bal_list.txt"
        lst.write_text("\n".join(sub) + "\n", encoding="utf-8")
        y = OUTDIR / f"aihub_val_{io}_bal.yaml"
        y.write_text(
            f"# AIHub val {io} 클래스통제(1:1:1, {per}×3={per*3}) — C4 재평가. 0=smoke,1=fire.\n"
            f"train: {lst.as_posix()}\nval: {lst.as_posix()}\n\nnames: ['smoke', 'fire']\nnc: 2\n",
            encoding="utf-8")
        r = model.val(data=str(y), imgsz=IMGSZ, device=DEVICE, split="val",
                      project=PROJECT, name=f"AIHub_C4_val{io}_bal", exist_ok=True, plots=False, verbose=False)
        results[io] = metrics(r)
        print(f"[{io} 균형] mAP50={results[io]['map50']} fireAP={results[io]['fire_ap50']} smokeAP={results[io]['smoke_ap50']}")

    (Path(PROJECT) / "aihub_val_inout_balanced.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    d = results["in"]["map50"] - results["out"]["map50"]
    print(f"\n===== 클래스통제 요약 =====")
    print(f"  실내={results['in']['map50']}  실외={results['out']['map50']}  Δ(실내-실외)={d:+.4f}")
    print(f"  (통제 전: 실내 0.946 vs 실외 0.824, Δ+0.122)")
    print("[저장] runs/aihub_val_inout_balanced.json")


if __name__ == "__main__":
    main()
