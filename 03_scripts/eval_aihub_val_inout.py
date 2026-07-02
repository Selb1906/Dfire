"""우선순위1 — AIHub C4(114K 학습) 체크포인트를 val 실내/실외 서브셋에 재평가(재학습 없음).

목적: 전체 val 0.913이 '실내 단독'인지 '실내+실외 평균'인지 규명 (건축물 비화재보 동기 직결).
방법: Validation 라벨 JSON에서 clip(prefix=clip_class_place)→inout 맵 구축 → val 이미지 버킷팅 →
      실내/실외 val 리스트 + yaml 생성 → AIHub_C4/best.pt로 각각 model.val(). 전체 val도 함께(sanity=0.913).
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
IMGSZ, DEVICE = 640, "0"


def prefix(stem):
    return "_".join(stem.split("_")[:3])   # clip_class_place (예: 0005_SM_GAH)


def build_inout_map():
    branch = next((VAL_LABELS / b for b in BRANCHES if (VAL_LABELS / b).exists()), None)
    print(f"Validation 브랜치: {branch}")
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
        io = (d.get("attributes", {}) or {}).get("inout")
        inout[prefix(Path(fn).stem)] = io
    print(f"clip→inout 맵: {len(inout)}개 (분포 {Counter(inout.values())})")
    return inout


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]
    return {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "smoke_ap50": round(ap50[0], 4) if ap50 else None,
            "fire_ap50": round(ap50[1], 4) if len(ap50) > 1 else None}


def write_yaml(name, listpath):
    y = OUTDIR / f"aihub_val_{name}.yaml"
    y.write_text(
        f"# AIHub val {name} 서브셋 (inout={name}). C4 재평가용. 0=smoke,1=fire.\n"
        f"train: {listpath.as_posix()}\n"   # ultralytics 파서 요구(더미)
        f"val: {listpath.as_posix()}\n\nnames: ['smoke', 'fire']\nnc: 2\n",
        encoding="utf-8")
    return str(y)


def main():
    from ultralytics import YOLO
    inout = build_inout_map()

    bucket = defaultdict(list)   # io → [경로]
    cls_by_io = defaultdict(Counter)
    unk = 0
    for p in sorted(VAL_IMG.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        io = inout.get(prefix(p.stem))
        cls = p.stem.split("_")[1] if len(p.stem.split("_")) > 1 else "?"
        if io in ("in", "out"):
            bucket[io].append(str(p)); cls_by_io[io][cls] += 1
        else:
            unk += 1
    print(f"val 버킷: 실내(in)={len(bucket['in'])} 실외(out)={len(bucket['out'])} unknown={unk}")
    for io in ("in", "out"):
        print(f"  [{io}] 클래스분포: {dict(cls_by_io[io])}")

    model = YOLO(C4)
    results = {"checkpoint": C4, "note": "AIHub C4(114K학습) → val inout 서브셋 재평가",
               "counts": {"in": len(bucket["in"]), "out": len(bucket["out"]), "unknown": unk}}
    # 전체 val (sanity=0.913)
    full_yaml = str(OUTDIR / "aihub_val.yaml")
    rf = model.val(data=full_yaml, imgsz=IMGSZ, device=DEVICE, split="val",
                   project=PROJECT, name="AIHub_C4_valfull", exist_ok=True, plots=False, verbose=False)
    results["full"] = metrics(rf)
    print(f"[full] mAP50={results['full']['map50']} (기대 0.913)")
    # 실내/실외
    for io in ("in", "out"):
        lst = OUTDIR / f"aihub_val_{io}_list.txt"
        lst.write_text("\n".join(bucket[io]) + "\n", encoding="utf-8")
        y = write_yaml(io, lst)
        r = model.val(data=y, imgsz=IMGSZ, device=DEVICE, split="val",
                      project=PROJECT, name=f"AIHub_C4_val{io}", exist_ok=True, plots=False, verbose=False)
        results[io] = metrics(r)
        print(f"[{io}] mAP50={results[io]['map50']} fireAP={results[io]['fire_ap50']} smokeAP={results[io]['smoke_ap50']}")

    (Path(PROJECT) / "aihub_val_inout.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== 요약 (AIHub C4 재평가) =====")
    print(f"  full={results['full']['map50']}  실내={results['in']['map50']}  실외={results['out']['map50']}")
    print("[저장] runs/aihub_val_inout.json")


if __name__ == "__main__":
    main()
