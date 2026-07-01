"""Design A — AIHub 실외(out)/실내(in) 서브셋 결합셋 구성 (naive와 동일 구조, inout만 변경).

naive: D-Fire ×3 + AIHub 혼합 14,122.  →  out/in: D-Fire ×3 + AIHub {실외|실내} 14,122.
AIHub 이미지의 inout은 clip 단위 속성 → JSON(Training_labels)에서 clip→inout 맵 구축 후,
yolo_071751 이미지 파일명(clip_class_place_frame)으로 조회해 버킷팅.
출력: compositions/combined_{out,in}_train.txt + yaml + 분포 리포트.
"""
from __future__ import annotations
from pathlib import Path
import json
from collections import Counter

LABELS = Path(r"D:\AIHub_Fire\extracted\Training_labels")
YOLO_TRAIN = Path(r"D:\AIHub_Fire\yolo_071751\images\train")
DFIRE_TRAIN = Path(r"C:\YangHyunHo\DFire\dfire\data\train\images")
OUTDIR = Path(r"C:\YangHyunHo\DFire\compositions")
BRANCHES = ["화재 현상", "화재현상"]
DFIRE_REPEAT = 3
N_SUB = 14122          # AIHub 서브셋 크기 (naive와 동일)
IMG_EXT = {".jpg", ".jpeg", ".png"}


def prefix(stem):
    return "_".join(stem.split("_")[:3])   # clip_class_place (예: 0087_FL_FWW)


def even_sample(items, k):
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def main():
    branch = next((LABELS / b for b in BRANCHES if (LABELS / b).exists()), None)
    # clip(prefix)→inout 맵 (clip당 1개면 충분)
    inout = {}
    seen = set()
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

    # yolo 이미지 버킷팅: (inout, class) 별로 수집
    from collections import defaultdict
    bucket = defaultdict(list)   # (io, cls) → [경로]
    unk = 0
    for p in sorted(YOLO_TRAIN.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        io = inout.get(prefix(p.stem))
        cls = p.stem.split("_")[1] if len(p.stem.split("_")) > 1 else "?"
        if io in ("out", "in"):
            bucket[(io, cls)].append(str(p))
        else:
            unk += 1
    for io in ("out", "in"):
        print(f"  실{'외' if io=='out' else '내'} 클래스: " +
              ", ".join(f"{c}={len(bucket[(io,c)])}" for c in ("SM", "FL", "NONE")))

    # 클래스 매칭 서브셋: 각 inout에서 SM:FL:NONE = 1:1:1 로 N_SUB 샘플 (inout만 변수)
    per = N_SUB // 3
    dfire = sorted(str(p) for p in DFIRE_TRAIN.iterdir() if p.suffix.lower() in IMG_EXT)
    for name in ("out", "in"):
        sub = (even_sample(bucket[(name, "SM")], per)
               + even_sample(bucket[(name, "FL")], per)
               + even_sample(bucket[(name, "NONE")], N_SUB - 2 * per))
        lines = dfire * DFIRE_REPEAT + sub
        (OUTDIR / f"combined_{name}_train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_df = len(dfire) * DFIRE_REPEAT
        tot = n_df + len(sub)
        (OUTDIR / f"combined_{name}.yaml").write_text(
            f"# Design A — D-Fire ×{DFIRE_REPEAT}({n_df}) + AIHub {'실외' if name=='out' else '실내'}({len(sub)}) = {tot}\n"
            f"# naive와 동일 구조, AIHub inout만 {name}. val/test=D-Fire. 0=smoke,1=fire.\n"
            f"train: {(OUTDIR / f'combined_{name}_train.txt').as_posix()}\n"
            f"val: C:/YangHyunHo/DFire/dfire/data/val/images\n"
            f"test: C:/YangHyunHo/DFire/dfire/data/test/images\n\nnames: ['smoke', 'fire']\nnc: 2\n",
            encoding="utf-8")
        print(f"  [{name}] D-Fire×{DFIRE_REPEAT} + AIHub-{name} {len(sub)} = {tot}")


if __name__ == "__main__":
    main()
