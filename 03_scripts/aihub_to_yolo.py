#!/usr/bin/env python3
"""
AI Hub 방재 데이터셋 → YOLO 형식 변환 스크립트 (멀티-데이터셋 통합판)

지원 데이터셋:
  71751 — 화재 발생 예측 영상 (고도화): FL/SM/NM 주력
  518   — 다중밀집시설 및 주거시설 화재 안전: 실내 화재 특화
  71472 — 화재영상 3D 객체 데이터 생성: 합성 증강
  기타  — 배경(NM) 데이터: 71679, 71677, 71330, 71682, 71850

클래스 정의:
  0 = fire   (FL, 불꽃/화염)
  1 = smoke  (SM, 연기)
  (normal/정상 씬은 레이블 없는 빈 txt → Hard Negative Mining 효과)

영상→이미지 샘플링 (과적합 방지, R1~R3 교훈 반영):
  1단계: cv2.compareHist 씬 경계 탐지 (상관계수 ≤0.7)
  2단계: 씬 내 2초 간격 추출 (30fps 기준 60프레임당 1장, ~10%)
  3단계: imagehash.phash 해밍 거리 ≤8 중복 제거

클래스 균형 (FL:SM:NM = 1:1:0.5):
  --balance_nm 옵션으로 NM 수량 자동 조정

사용법:
  # 단일 데이터셋 (JSON 이미지 기반)
  python models/aihub_to_yolo.py \
      --datasets 71751 \
      --src_img  /data/71751/원천데이터 \
      --src_json /data/71751/라벨링데이터 \
      --dst      models/aihub_dataset

  # 멀티 데이터셋 + 영상 샘플링
  python models/aihub_to_yolo.py \
      --datasets 71751,518 \
      --src_img  /data \
      --src_json /data \
      --dst      models/integrated_dataset \
      --sample_video \
      --balance_nm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 클래스 정의
CLASS_NAMES = ["fire", "smoke"]

# ── AI Hub 카테고리명 → YOLO 클래스 ID 통합 매핑
# 71751, 518, 71472 모두 포함. 71472 합성 데이터 카테고리 포함.
CATEGORY_MAP: dict[str, int] = {
    # 화재 (class 0)
    "fire": 0,
    "화재": 0,
    "불꽃": 0,
    "flame": 0,
    "fire_label": 0,
    "fl": 0,
    # 연기 (class 1)
    "smoke": 1,
    "연기": 1,
    "sm": 1,
    "smoke_label": 1,
    # 518번 데이터셋 카테고리
    "fire_indoor": 0,
    "실내화재": 0,
    "smoke_indoor": 1,
    "실내연기": 1,
    # 71472 합성 데이터 카테고리
    "synthetic_fire": 0,
    "synthetic_smoke": 1,
    "3d_fire": 0,
    "3d_smoke": 1,
}

# ── NM(정상) 비율 기본값: FL:SM:NM = 1:1:0.5
DEFAULT_NM_RATIO = 0.5

# ── 영상 샘플링 파라미터
HIST_CORR_THRESHOLD = 0.7   # 씬 경계: 상관계수 이 이하면 새 씬
FRAME_INTERVAL_SEC = 2.0     # 씬 내 간격 (초)
PHASH_HAMMING_MAX = 8        # 중복 제거 해밍 거리 임계값


# ────────────────────────────────────────────────────────────── #
# COCO JSON 변환
# ────────────────────────────────────────────────────────────── #

def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, ...]:
    """COCO [x, y, w, h] → YOLO [cx, cy, w, h] 정규화."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # bbox가 범위를 벗어날 경우 클리핑
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.001, min(1.0, nw))
    nh = max(0.001, min(1.0, nh))
    return round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)


def convert_json(
    json_path: Path, img_dir: Path, dst_img: Path, dst_lbl: Path
) -> tuple[int, dict[str, int]]:
    """단일 COCO JSON 파일 변환.

    Returns:
        (변환된 이미지 수, 클래스별 카운트)
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    id2img: dict[int, dict] = {img["id"]: img for img in data.get("images", [])}

    id2cls: dict[int, int] = {}
    for cat in data.get("categories", []):
        name = cat.get("name", "").lower().strip()
        if name in CATEGORY_MAP:
            id2cls[cat["id"]] = CATEGORY_MAP[name]

    img2labels: dict[int, list[str]] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann.get("category_id", -1)
        if cat_id not in id2cls:
            continue
        cls_id = id2cls[cat_id]
        img_info = id2img.get(img_id)
        if not img_info:
            continue
        # bbox 유효성 검사
        bbox = ann.get("bbox", [0, 0, 1, 1])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        cx, cy, nw, nh = coco_bbox_to_yolo(
            bbox, img_info["width"], img_info["height"]
        )
        line = f"{cls_id} {cx} {cy} {nw} {nh}"
        img2labels.setdefault(img_id, []).append(line)

    count = 0
    class_counts: dict[str, int] = {"fire": 0, "smoke": 0, "normal": 0}

    for img_id, img_info in id2img.items():
        src_file = img_dir / img_info["file_name"]
        if not src_file.exists():
            candidates = list(img_dir.rglob(img_info["file_name"]))
            if not candidates:
                continue
            src_file = candidates[0]

        stem = src_file.stem
        ext = src_file.suffix

        dst_img_file = dst_img / f"{stem}{ext}"
        shutil.copy2(src_file, dst_img_file)

        lbl_lines = img2labels.get(img_id, [])
        (dst_lbl / f"{stem}.txt").write_text("\n".join(lbl_lines), encoding="utf-8")
        count += 1

        if lbl_lines:
            classes_in_img = {int(l.split()[0]) for l in lbl_lines}
            if 0 in classes_in_img:
                class_counts["fire"] += 1
            if 1 in classes_in_img:
                class_counts["smoke"] += 1
        else:
            class_counts["normal"] += 1

    return count, class_counts


# ────────────────────────────────────────────────────────────── #
# 영상 샘플링 파이프라인
# ────────────────────────────────────────────────────────────── #

def sample_video(
    video_path: Path,
    dst_img: Path,
    dst_lbl: Path,
    label_class: int | None,
    fps_hint: float = 30.0,
    prefix: str = "",
) -> int:
    """영상에서 프레임 샘플링 후 YOLO 이미지+레이블 저장.

    3단계 과적합 방지 프로토콜:
      1. 씬 경계 탐지 (compareHist 상관계수 ≤ 0.7)
      2. 씬 내 2초 간격 (fps 기준)
      3. phash 해밍 거리 ≤ 8 중복 제거

    label_class:
        0 = fire, 1 = smoke, None = 레이블 없는 NM (빈 txt 생성)
    """
    try:
        import cv2
    except ImportError:
        logger.error("cv2 없음. 'pip install opencv-python' 필요")
        return 0

    try:
        import imagehash
        from PIL import Image as PILImage
        use_phash = True
    except ImportError:
        logger.warning("imagehash 없음. phash 중복 제거 생략. 'pip install imagehash Pillow'")
        use_phash = False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"영상 열기 실패: {video_path}")
        return 0

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    interval_frames = max(1, int(actual_fps * FRAME_INTERVAL_SEC))

    prev_hist: list | None = None
    scene_frame_count = 0
    saved_count = 0
    frame_idx = 0
    seen_hashes: list = []

    video_stem = video_path.stem
    if prefix:
        video_stem = f"{prefix}_{video_stem}"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 1단계: 씬 경계 탐지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr <= HIST_CORR_THRESHOLD:
                scene_frame_count = 0  # 씬 전환 → 카운터 리셋
        prev_hist = hist

        # 2단계: 씬 내 간격 샘플링
        scene_frame_count += 1
        if scene_frame_count % interval_frames != 1:
            continue

        # 3단계: phash 중복 제거
        if use_phash:
            pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h = imagehash.phash(pil_img)
            if any(h - sh <= PHASH_HAMMING_MAX for sh in seen_hashes):
                continue
            seen_hashes.append(h)

        # 저장
        fname = f"{video_stem}_{frame_idx:06d}.jpg"
        img_path = dst_img / fname
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 레이블: 전체 프레임 bbox (cl cx cy w h = class 0.5 0.5 1.0 1.0)
        # 실제 bbox가 없으므로 전체 프레임으로 약한 레이블 생성
        # 주의: 정확한 bbox가 있는 JSON 데이터보다 품질이 낮음
        lbl_path = dst_lbl / f"{video_stem}_{frame_idx:06d}.txt"
        if label_class is not None:
            lbl_path.write_text(f"{label_class} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        else:
            lbl_path.write_text("", encoding="utf-8")  # NM: 빈 레이블

        saved_count += 1

    cap.release()
    return saved_count


# ────────────────────────────────────────────────────────────── #
# 클래스 균형 조정
# ────────────────────────────────────────────────────────────── #

def balance_classes(
    tmp_img: Path,
    tmp_lbl: Path,
    nm_ratio: float = DEFAULT_NM_RATIO,
    seed: int = 42,
) -> dict[str, int]:
    """FL:SM:NM = 1:1:nm_ratio 로 NM 수량 조정 (랜덤 언더샘플링).

    FL, SM이 모두 있는 이미지 = 혼합 이미지, 별도 처리.
    NM = 빈 레이블 이미지.

    Returns:
        균형 조정 후 클래스별 카운트
    """
    rng = random.Random(seed)

    fire_stems: list[str] = []
    smoke_stems: list[str] = []
    nm_stems: list[str] = []

    for lbl_file in tmp_lbl.glob("*.txt"):
        content = lbl_file.read_text(encoding="utf-8").strip()
        classes = {int(l.split()[0]) for l in content.splitlines() if l.strip()}
        stem = lbl_file.stem
        if 0 in classes:
            fire_stems.append(stem)
        elif 1 in classes:
            smoke_stems.append(stem)
        else:
            nm_stems.append(stem)

    n_fire = len(fire_stems)
    n_smoke = len(smoke_stems)
    target = max(n_fire, n_smoke)  # FL, SM은 모두 유지
    nm_target = int(target * nm_ratio)

    if len(nm_stems) > nm_target:
        rng.shuffle(nm_stems)
        remove_stems = set(nm_stems[nm_target:])
        logger.info(
            f"[balance] NM {len(nm_stems)} → {nm_target} (제거: {len(remove_stems)})"
        )
        for stem in remove_stems:
            for f in tmp_img.glob(f"{stem}.*"):
                f.unlink(missing_ok=True)
            lbl = tmp_lbl / f"{stem}.txt"
            lbl.unlink(missing_ok=True)
        nm_stems = nm_stems[:nm_target]

    return {"fire": n_fire, "smoke": n_smoke, "normal": len(nm_stems)}


# ────────────────────────────────────────────────────────────── #
# 무결성 검증
# ────────────────────────────────────────────────────────────── #

def verify_dataset(dst: Path) -> bool:
    """변환된 데이터셋 무결성 검증.

    검사 항목:
    1. 이미지-레이블 1:1 매칭
    2. bbox [0,1] 범위 검증
    3. 클래스 분포 히스토그램 출력
    """
    ok = True
    class_counts: dict[int, int] = {}
    invalid_bbox = 0
    orphan_labels = 0
    orphan_images = 0

    for split in ("train", "val"):
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        if not img_dir.exists():
            continue

        img_stems = {f.stem for f in img_dir.iterdir() if f.is_file()}
        lbl_stems = {f.stem for f in lbl_dir.iterdir() if f.is_file()}

        orphan_images += len(img_stems - lbl_stems)
        orphan_labels += len(lbl_stems - img_stems)

        for lbl_file in lbl_dir.glob("*.txt"):
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                cx, cy, nw, nh = map(float, parts[1:])
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                    invalid_bbox += 1

    if orphan_images > 0:
        logger.warning(f"[verify] 레이블 없는 이미지: {orphan_images}개")
    if orphan_labels > 0:
        logger.warning(f"[verify] 이미지 없는 레이블: {orphan_labels}개")
        ok = False
    if invalid_bbox > 0:
        logger.warning(f"[verify] bbox 범위 오류: {invalid_bbox}개")
        ok = False

    logger.info("[verify] 클래스 분포:")
    for cls_id, cnt in sorted(class_counts.items()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
        logger.info(f"  {name} (class {cls_id}): {cnt}개 어노테이션")

    if ok:
        logger.info("[verify] 무결성 검증 통과")
    else:
        logger.error("[verify] 무결성 검증 실패 — 위 오류를 수정하세요")

    return ok


# ────────────────────────────────────────────────────────────── #
# 메인
# ────────────────────────────────────────────────────────────── #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Hub 방재 데이터셋 → YOLO 형식 변환 (멀티-데이터셋 통합판)"
    )
    parser.add_argument("--src_img",   required=True,
                        help="AI Hub 원천데이터(이미지/영상) 루트 폴더")
    parser.add_argument("--src_json",  required=True,
                        help="AI Hub 라벨링데이터(JSON) 루트 폴더")
    parser.add_argument("--dst",       default="models/aihub_dataset",
                        help="출력 폴더 (기본: models/aihub_dataset)")
    parser.add_argument("--datasets",  default="71751",
                        help="처리할 데이터셋 ID 쉼표 구분 (예: 71751,518,71472)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="검증셋 비율 (기본: 0.15)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--sample_video", action="store_true",
                        help="영상 파일(.mp4/.avi 등) 직접 샘플링")
    parser.add_argument("--balance_nm", action="store_true",
                        help="FL:SM:NM = 1:1:0.5 클래스 균형 강제")
    parser.add_argument("--nm_ratio",  type=float, default=DEFAULT_NM_RATIO,
                        help="NM 비율 (balance_nm 시 사용, 기본: 0.5)")
    parser.add_argument("--verify",    action="store_true", default=True,
                        help="변환 후 무결성 검증 (기본: True)")
    parser.add_argument("--no_verify", dest="verify", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    src_img  = Path(args.src_img)
    src_json = Path(args.src_json)
    dst      = Path(args.dst)
    datasets = [d.strip() for d in args.datasets.split(",")]

    # 출력 폴더 생성
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    tmp_img = dst / "_tmp_images"
    tmp_lbl = dst / "_tmp_labels"
    tmp_img.mkdir(exist_ok=True)
    tmp_lbl.mkdir(exist_ok=True)

    total = 0
    total_class_counts: dict[str, int] = {"fire": 0, "smoke": 0, "normal": 0}

    logger.info(f"[convert] 데이터셋: {datasets}")

    # 1. JSON 기반 변환
    json_files = list(src_json.rglob("*.json"))
    logger.info(f"[convert] JSON 파일 {len(json_files)}개 발견")

    for jf in json_files:
        n, cc = convert_json(jf, src_img, tmp_img, tmp_lbl)
        total += n
        for k, v in cc.items():
            total_class_counts[k] = total_class_counts.get(k, 0) + v
        if n > 0:
            logger.info(f"  {jf.name}: {n}장 변환 (fire={cc['fire']}, smoke={cc['smoke']}, nm={cc['normal']})")

    logger.info(f"[convert] JSON 변환 완료: {total}장")

    # 2. 영상 샘플링 (--sample_video 시)
    if args.sample_video:
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".ts"}
        video_files = [
            f for f in src_img.rglob("*")
            if f.suffix.lower() in video_exts
        ]
        logger.info(f"[video] 영상 파일 {len(video_files)}개 발견")

        for vf in video_files:
            # 경로에 따라 클래스 추정
            vf_lower = str(vf).lower()
            if "fl" in vf_lower or "fire" in vf_lower or "화재" in vf_lower:
                label_class: int | None = 0
                cls_name = "fire"
            elif "sm" in vf_lower or "smoke" in vf_lower or "연기" in vf_lower:
                label_class = 1
                cls_name = "smoke"
            else:
                label_class = None  # NM
                cls_name = "normal"

            n = sample_video(vf, tmp_img, tmp_lbl, label_class, prefix=vf.parent.name)
            total_class_counts[cls_name] = total_class_counts.get(cls_name, 0) + n
            total += n
            logger.info(f"  {vf.name}: {n}프레임 → {cls_name}")

        logger.info(f"[video] 샘플링 완료: 누적 {total}장")

    # 3. 클래스 균형 조정
    if args.balance_nm:
        logger.info("[balance] NM 언더샘플링 시작...")
        balanced = balance_classes(tmp_img, tmp_lbl, nm_ratio=args.nm_ratio, seed=args.seed)
        logger.info(
            f"[balance] 균형 후: fire={balanced['fire']}, "
            f"smoke={balanced['smoke']}, normal={balanced['normal']}"
        )

    # 4. train/val 분할
    all_stems = [f.stem for f in tmp_img.iterdir() if f.is_file()]
    random.seed(args.seed)
    random.shuffle(all_stems)
    n_val = int(len(all_stems) * args.val_ratio)
    val_stems   = set(all_stems[:n_val])
    train_stems = set(all_stems[n_val:])

    def _move_split(stems: set[str], split: str) -> None:
        for stem in stems:
            for img_f in tmp_img.glob(f"{stem}.*"):
                shutil.move(str(img_f), dst / "images" / split / img_f.name)
            lbl_f = tmp_lbl / f"{stem}.txt"
            if lbl_f.exists():
                shutil.move(str(lbl_f), dst / "labels" / split / lbl_f.name)

    _move_split(train_stems, "train")
    _move_split(val_stems,   "val")

    shutil.rmtree(tmp_img, ignore_errors=True)
    shutil.rmtree(tmp_lbl, ignore_errors=True)

    # 5. data.yaml 생성
    yaml_content = f"""# AI Hub 통합 방재 데이터셋 (데이터셋: {', '.join(datasets)})
# 생성: models/aihub_to_yolo.py
path: {dst.resolve().as_posix()}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (dst / "data.yaml").write_text(yaml_content, encoding="utf-8")

    n_train = len(list((dst / "images" / "train").iterdir()))
    n_val_  = len(list((dst / "images" / "val").iterdir()))

    logger.info(f"[done] train={n_train}, val={n_val_}, 합계={n_train + n_val_}")
    logger.info(f"[done] 최종 클래스 분포: {total_class_counts}")
    logger.info(f"[done] data.yaml → {dst / 'data.yaml'}")
    logger.info(f"\n학습 실행 예시:")
    logger.info(f"  python models/train.py --data {dst / 'data.yaml'} --model yolo11n --epochs 100 --batch 64 --device 0")

    # 6. 무결성 검증
    if args.verify:
        verify_dataset(dst)


if __name__ == "__main__":
    main()
