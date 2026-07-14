"""혼합물설계 Phase A — 순수 단일라벨 풀 독립 재카운트 + 샘플링 인덱스 구축.

라벨(0=smoke,1=fire) 기준으로 train 이미지를 fire_only/smoke_only/both/nm 분류(both=동시라벨, 설계상 제외).
inout(실내/실외)은 Training_labels JSON의 clip→inout 맵에서. 클립=파일명 prefix(clip_class_place).
출력: runs/mixture_pool_index.json  = {component: {inout: {clip: [img_paths]}}} + counts.
목적: ① 28,376=28,376 검증(원본 라벨 독립 카운트) ② 샘플러가 쓸 인덱스(재읽기 방지).
"""
from __future__ import annotations
import json, os
from pathlib import Path
from collections import Counter, defaultdict

TR = Path(r"D:\AIHub_Fire\yolo_071751")
LABELS = Path(r"D:\AIHub_Fire\extracted\Training_labels")
BRANCHES = ["화재 현상", "화재현상"]
OUT = Path(r"C:\YangHyunHo\DFire\runs\mixture_pool_index.json")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def prefix(stem):
    return "_".join(stem.split("_")[:3])   # clip_class_place


def build_inout_map():
    branch = next((LABELS / b for b in BRANCHES if (LABELS / b).exists()), None)
    m, seen = {}, set()
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
        m[prefix(Path(fn).stem)] = (d.get("attributes", {}) or {}).get("inout")
    return m


def classify(lbl: Path):
    try:
        lines = [l for l in lbl.read_text().splitlines() if l.strip()]
    except Exception:
        return "nm"
    if not lines:
        return "nm"
    cls = set(int(l.split()[0]) for l in lines)
    hs, hf = 0 in cls, 1 in cls
    if hf and hs: return "both"
    if hf: return "fire_only"
    if hs: return "smoke_only"
    return "nm"


def main():
    inout = build_inout_map()
    print(f"clip→inout 맵 {len(inout)} (분포 {Counter(inout.values())})", flush=True)
    imgs, lbls = TR / "images" / "train", TR / "labels" / "train"
    # pool[component][inout][clip] = [img_path]
    pool = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cnt = defaultdict(Counter)   # component → inout Counter
    unk_io = 0
    for p in sorted(imgs.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        comp = classify(lbls / (p.stem + ".txt"))
        if comp == "both":
            cnt["both"]["-"] += 1
            continue   # 동시라벨 제외
        io = inout.get(prefix(p.stem))
        if io not in ("in", "out"):
            io = "unknown"; unk_io += 1
        pool[comp][io][prefix(p.stem)].append(str(p))
        cnt[comp][io] += 1
    # 리포트
    print("\n=== 순수 풀 독립 재카운트 (라벨 기준) ===")
    for comp in ("fire_only", "smoke_only", "nm", "both"):
        tot = sum(cnt[comp].values())
        byio = dict(cnt[comp])
        print(f"  {comp:11s}: {tot:>7,}   inout={byio}")
    print(f"  unknown inout: {unk_io}")
    # inout 비율(층화용)
    print("\n=== 성분별 inout 비율(층화 기준) ===")
    for comp in ("fire_only", "smoke_only", "nm"):
        i, o = cnt[comp]["in"], cnt[comp]["out"]
        s = i + o
        print(f"  {comp}: in={i}({i/s*100:.1f}%) out={o}({o/s*100:.1f}%)  clips={sum(len(pool[comp][x]) for x in ('in','out','unknown'))}")
    # 저장 (dict화)
    idx = {c: {io: {clip: fs for clip, fs in cl.items()} for io, cl in pool[c].items()} for c in pool}
    OUT.write_text(json.dumps({"counts": {c: dict(cnt[c]) for c in cnt}, "pool": idx}, ensure_ascii=False), encoding="utf-8")
    print(f"\n[저장] {OUT}  (N=6000 대비 여유: fire {sum(cnt['fire_only'].values())}, smoke {sum(cnt['smoke_only'].values())}, nm {sum(cnt['nm'].values())})", flush=True)


if __name__ == "__main__":
    main()
