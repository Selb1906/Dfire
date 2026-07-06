"""AIHub held-out test — val 19,080(530클립)을 **클립 단위** 60/40으로 분할, 기존 체크포인트 재평가(재학습 없음).

누수방지: 같은 클립(원본영상)의 인접 프레임이 val'/test 양쪽에 걸치지 않게 클립 ID로 분할(클래스 층화, seed=0).
목적: val 기준 헤드라인 vs 진짜 held-out test 차이 확인(§6.3 한계 보강). 결과 → runs/heldout_test.json.
평가: 멀티시드 3체크포인트(s0=C4/s1/s2)를 full/실내/실외/실내균형/실외균형에, C1/C2/C3를 full에.
"""
from __future__ import annotations
import json, os, random, statistics as st
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
VAL_IMG = Path(r"D:\AIHub_Fire\yolo_071751\images\val")
VAL_LABELS = Path(r"D:\AIHub_Fire\extracted\Validation_labels")
BRANCHES = ["화재 현상", "화재현상"]
COMP = BASE / "compositions"
TEST_FRAC = 0.40
CLASSES = ("SM", "FL", "NONE")
IMGSZ, DEVICE = 640, "0"
MULTI = [("s0", "AIHub_C4"), ("s1", "AIHub_C4_s1"), ("s2", "AIHub_C4_s2")]
CSER = [("C1", "AIHub_C1"), ("C2", "AIHub_C2"), ("C3", "AIHub_C3")]


def prefix(stem): return "_".join(stem.split("_")[:3])
def even_sample(items, k):
    if k >= len(items): return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def build_inout_map():
    branch = next((VAL_LABELS / b for b in BRANCHES if (VAL_LABELS / b).exists()), None)
    m, seen = {}, set()
    for jp in branch.rglob("JSON/*.json"):
        clip = jp.parent.parent.name
        if clip in seen: continue
        seen.add(clip)
        try: d = json.load(open(jp, encoding="utf-8"))
        except Exception: continue
        fn = d.get("image", {}).get("filename", jp.stem + ".jpg")
        m[prefix(Path(fn).stem)] = (d.get("attributes", {}) or {}).get("inout")
    return m


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]
    return {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "smoke_ap50": round(ap50[0], 4) if ap50 else None,
            "fire_ap50": round(ap50[1], 4) if len(ap50) > 1 else None}


def write_yaml(name, imgs):
    lst = COMP / f"heldout_{name}_list.txt"
    lst.write_text("\n".join(imgs) + "\n", encoding="utf-8")
    y = COMP / f"heldout_{name}.yaml"
    y.write_text(f"# held-out test {name} ({len(imgs)}장, 클립단위 분할). 0=smoke,1=fire.\n"
                 f"train: {lst.as_posix()}\nval: {lst.as_posix()}\n\nnames: ['smoke', 'fire']\nnc: 2\n",
                 encoding="utf-8")
    return str(y)


def ms(vals):
    return round(st.mean(vals), 4), (round(st.stdev(vals), 4) if len(vals) > 1 else 0.0)


def main():
    from ultralytics import YOLO
    inout = build_inout_map()
    # 클립 인덱싱
    clip_frames, clip_cls, clip_io = defaultdict(list), {}, {}
    for p in sorted(VAL_IMG.iterdir()):
        if p.suffix.lower() != ".jpg": continue
        st_ = p.stem; clip = st_.split("_")[0]
        clip_frames[clip].append(str(p))
        clip_cls[clip] = st_.split("_")[1] if len(st_.split("_")) > 1 else "?"
        clip_io[clip] = inout.get(prefix(st_))
    # 클래스 층화 클립 분할(seed=0)
    rng = random.Random(0)
    by_cls = defaultdict(list)
    for c in clip_frames: by_cls[clip_cls[c]].append(c)
    test_clips = set()
    for cls, clips in by_cls.items():
        clips = sorted(clips); rng.shuffle(clips)
        test_clips.update(clips[:round(len(clips) * TEST_FRAC)])
    # held-out test 프레임 버킷
    test_all, test_io, test_iocls = [], defaultdict(list), defaultdict(list)
    for clip, frames in clip_frames.items():
        if clip not in test_clips: continue
        test_all += frames
        io, cls = clip_io[clip], clip_cls[clip]
        if io in ("in", "out"):
            test_io[io] += frames
            if cls in CLASSES: test_iocls[(io, cls)] += frames
    per = min(len(test_iocls[(io, c)]) for io in ("in", "out") for c in CLASSES)
    print(f"클립 분할: 전체 {len(clip_frames)} → test {len(test_clips)} 클립 / {len(test_all)} 프레임 "
          f"(val' {len(clip_frames)-len(test_clips)}클립)")
    print(f"  test 실내 {len(test_io['in'])} / 실외 {len(test_io['out'])} / 균형 per-class {per}")

    yml = {"full": write_yaml("full", test_all),
           "in": write_yaml("in", test_io["in"]), "out": write_yaml("out", test_io["out"])}
    for io in ("in", "out"):
        bal = []
        for c in CLASSES: bal += even_sample(test_iocls[(io, c)], per)
        yml[f"{io}_bal"] = write_yaml(f"{io}_bal", bal)

    def val_ckpt(rel, y):
        bp = Path(PROJECT) / rel / "weights" / "best.pt"
        r = YOLO(str(bp)).val(data=y, imgsz=IMGSZ, device=DEVICE, split="val",
                              project=PROJECT, name=f"{rel}_heldout", exist_ok=True, plots=False, verbose=False)
        return metrics(r)

    results = {"test_clips": len(test_clips), "test_frames": len(test_all), "test_frac": TEST_FRAC,
               "split": "clip-level stratified (seed=0)", "multiseed": {}, "cseries": {}}
    # 멀티시드 3체크포인트 × 5셋 → mean±std
    for tag in ("full", "in", "out", "in_bal", "out_bal"):
        per_seed = {s: val_ckpt(rel, yml[tag]) for s, rel in MULTI}
        m, sd = ms([per_seed[s]["map50"] for s, _ in MULTI])
        fm, _ = ms([per_seed[s]["fire_ap50"] for s, _ in MULTI])
        sm, _ = ms([per_seed[s]["smoke_ap50"] for s, _ in MULTI])
        results["multiseed"][tag] = {"map50_mean": m, "map50_std": sd, "fire_mean": fm,
                                     "smoke_mean": sm, "per_seed": per_seed}
        print(f"[test {tag}] mAP50={m}±{sd}")
    # C1/C2/C3 × full
    for cn, rel in CSER:
        results["cseries"][cn] = val_ckpt(rel, yml["full"])
        print(f"[test {cn}] mAP50={results['cseries'][cn]['map50']}")

    (Path(PROJECT) / "heldout_test.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[저장] runs/heldout_test.json")


if __name__ == "__main__":
    main()
