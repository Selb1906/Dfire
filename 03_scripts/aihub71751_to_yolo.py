"""AIHub 071751 (화재 발생 예측 영상 고도화) per-image JSON → YOLO 변환.

★ 상태: 2026-06-24 초안 — 원천 해제 완료 후 end-to-end 검증 필요(아래 ASSUMPTIONS 참조).
기존 aihub_to_yolo.py(COCO 통짜 가정)는 이 데이터에 안 맞아 071751 전용으로 신규 작성.

데이터 구조 (D:/AIHub_Fire/extracted):
  Training_source/화재현상/이미지/{불꽃|연기|...}/{씬타입}/{clip}/JPG/{clip}_..._NNNNN.jpg
  Training_labels/화재 현상/이미지/{불꽃|연기|...}/{씬타입}/{clip}/JSON/{clip}_..._NNNNN.json
  (Validation_* 동일 구조)
  ※ 원천=`화재현상`(공백X), 라벨=`화재 현상`(공백O). JPG↔JSON, 확장자만 다름, stem 동일.

JSON 스키마:
  image{width,height,filename}
  attributes{class:"FL"/"SM"/..., clipname, scene, fps, ...}
  categories[{1:"fl"},{2:"sm"},{3:"none"}]
  annotations[{categories_id, bbox:[x,y,w,h]}]   # bbox = COCO식 좌상단+wh

클래스 매핑 (DFire 규약 통일): categories_id 1(fl)→1(fire), 2(sm)→0(smoke), 3(none)→무시.
출력 라벨: nc=2, names=['smoke','fire'] (0=smoke,1=fire).

샘플링: clip 단위로 frame_step 간격 추출(과적합/중복 방지). 1.9M 프레임 → 수십만으로 축소.
균형: --balance 시 FL:SM:NM 이미지 수를 1:1:nm_ratio 로 언더샘플.

ASSUMPTIONS (검증 TODO — 원천 해제 후):
  (A) 이미지 경로 = 라벨 경로에서 [labels→source, "화재 현상"→"화재현상", JSON→JPG, .json→.jpg].
  (B) NM(정상배경) = annotations 가 비었거나 categories_id 3(none)만 있는 이미지.
      별도 '정상' 폴더가 있으면 그 경로도 포함하도록 수정 필요.
  (C) "화재 현상" 브랜치만 사용. "화재현장 주요객체" 등 서브객체 브랜치는 제외(화재/연기 탐지 무관).
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["smoke", "fire"]            # 0=smoke, 1=fire
# 이름 기반 매핑(파일별 categories 배열 사용 — categories_id 순서 가정 제거).
NAME_TO_YOLO = {"fl": 1, "fire": 1, "sm": 0, "smoke": 0}  # none/기타 → 무시
# 브랜치명이 split마다 다름: Training=`화재 현상`(공백O), Validation=`화재현상`(공백X)
FIRE_BRANCHES = ["화재 현상", "화재현상"]
FIRE_BRANCH = "화재 현상"                   # label_to_image 경로치환용(공백→제거)


def find_branch(labels_root: Path):
    for b in FIRE_BRANCHES:
        if (labels_root / b).exists():
            return labels_root / b
    return None


def bbox_xywh_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw, nh = bw / w, bh / h
    cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
    nw, nh = min(max(nw, 1e-3), 1), min(max(nh, 1e-3), 1)
    return round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)


def label_to_image(json_path: Path, label_branch: Path, source_branch: Path) -> Path:
    """라벨 JSON → 원천 이미지 경로. 브랜치명 가정 없이 '브랜치 이후 상대경로'를 원천 브랜치에 이어붙임.
    (Training/Validation에서 `화재 현상`↔`화재현상` 공백 유무가 라벨·원천 간 교차하므로 이 방식이 안전.)"""
    rel = json_path.relative_to(label_branch)          # 이미지/불꽃/.../JSON/x.json
    parts = ["JPG" if p == "JSON" else p for p in rel.parts]
    return (source_branch / Path(*parts)).with_suffix(".jpg")


def parse_json(json_path: Path):
    """JSON → (yolo_lines, scene_class). yolo_lines 빈 리스트면 NM."""
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    img = d.get("image", {})
    w, h = img.get("width"), img.get("height")
    if not w or not h:
        return None, None
    # 파일별 categories: index → name (전역 순서 가정 제거)
    id2name = {c["category_index"]: str(c.get("category_name", "")).lower()
               for c in d.get("categories", [])}
    lines = []
    for ann in d.get("annotations", []):
        name = id2name.get(ann.get("categories_id"), "")
        if name not in NAME_TO_YOLO:        # none/기타 무시
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        cls = NAME_TO_YOLO[name]
        cx, cy, nw, nh = bbox_xywh_to_yolo(bbox, w, h)
        lines.append(f"{cls} {cx} {cy} {nw} {nh}")
    scene_cls = (d.get("attributes", {}) or {}).get("class")
    return lines, scene_cls


def collect(branch: Path):
    """화재(현상) 브랜치의 JSON 경로 수집 (clip별 그룹)."""
    by_clip = defaultdict(list)
    for jp in branch.rglob("JSON/*.json"):
        # clip = 부모의 부모 (……/{clip}/JSON/x.json)
        clip = jp.parent.parent.name
        by_clip[clip].append(jp)
    for clip in by_clip:
        by_clip[clip].sort()
    return by_clip


def category_of(lines):
    cls = {int(l.split()[0]) for l in lines}
    if not cls:
        return "nm"
    if cls == {1}:
        return "fire"
    if cls == {0}:
        return "smoke"
    return "both"


def convert_split(labels_root, source_root, dst, split, frame_step, rng):
    img_out = dst / "images" / split
    lbl_out = dst / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    label_branch = find_branch(labels_root)
    source_branch = find_branch(source_root)
    if label_branch is None or source_branch is None:
        logger.warning(f"[{split}] 브랜치 없음 (label={label_branch}, source={source_branch})")
        return img_out, lbl_out, {}

    by_clip = collect(label_branch)
    logger.info(f"[{split}] clip {len(by_clip)}개 (label={label_branch.name}, source={source_branch.name})")
    buckets = defaultdict(list)   # category → [(img_path, lines, stem)]
    n_missing = 0

    for clip, jsons in by_clip.items():
        for i, jp in enumerate(jsons):
            if i % frame_step != 0:          # clip 단위 프레임 간격 샘플링
                continue
            lines, _ = parse_json(jp)
            if lines is None:
                continue
            img_path = label_to_image(jp, label_branch, source_branch)
            if not img_path.exists():
                n_missing += 1
                continue
            buckets[category_of(lines)].append((img_path, lines, jp.stem))

    if n_missing:
        logger.warning(f"[{split}] 이미지 매칭 실패: {n_missing}장 (ASSUMPTION A 확인)")
    for k, v in buckets.items():
        logger.info(f"[{split}] {k}: {len(v)}장 (샘플링 후)")
    return img_out, lbl_out, buckets


def write_items(items, img_out, lbl_out):
    for img_path, lines, stem in items:
        dst_img = img_out / f"{stem}.jpg"
        if not dst_img.exists():
            try:
                os.link(img_path, dst_img)
            except OSError:
                shutil.copy2(img_path, dst_img)
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def balance(buckets, nm_ratio, rng):
    """FL:SM:NM = 1:1:nm_ratio. both 는 fire/smoke 양쪽 기여로 전량 유지."""
    fire = buckets.get("fire", []); smoke = buckets.get("smoke", [])
    both = buckets.get("both", []); nm = buckets.get("nm", [])
    target = max(len(fire) + len(both), len(smoke) + len(both))
    nm_target = int(target * nm_ratio)
    if len(nm) > nm_target:
        rng.shuffle(nm); nm = nm[:nm_target]
    return fire + smoke + both + nm


def main():
    ap = argparse.ArgumentParser(description="AIHub 071751 → YOLO (per-image JSON)")
    ap.add_argument("--src_root", default=r"D:\AIHub_Fire\extracted",
                    help="Training_*/Validation_* 가 있는 루트")
    ap.add_argument("--dst", default=r"D:\AIHub_Fire\yolo_071751")
    ap.add_argument("--frame_step", type=int, default=10,
                    help="clip 내 N프레임당 1장 (1.9M→축소)")
    ap.add_argument("--balance", action="store_true", help="FL:SM:NM 균형")
    ap.add_argument("--nm_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src_root); dst = Path(args.dst)

    for split, lab, srcd in [("train", "Training_labels", "Training_source"),
                             ("val", "Validation_labels", "Validation_source")]:
        labels_root = src / lab
        source_root = src / srcd
        if not labels_root.exists():
            logger.warning(f"{labels_root} 없음 → {split} 건너뜀")
            continue
        img_out, lbl_out, buckets = convert_split(
            labels_root, source_root, dst, split, args.frame_step, rng)
        items = balance(buckets, args.nm_ratio, rng) if args.balance \
            else [x for v in buckets.values() for x in v]
        logger.info(f"[{split}] 최종 {len(items)}장 기록 중...")
        write_items(items, img_out, lbl_out)

    yaml = (f"# AIHub 071751 → YOLO (0=smoke, 1=fire)\n"
            f"path: {dst.as_posix()}\n"
            f"train: images/train\nval: images/val\n\n"
            f"nc: 2\nnames: {CLASS_NAMES}\n")
    (dst / "data.yaml").write_text(yaml, encoding="utf-8")
    logger.info(f"[done] data.yaml → {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
