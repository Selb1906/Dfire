"""DFire 4셀 ablation 구성 빌드.

설계 (확정):
  - 평가셋 고정: val/test 는 전체 DFire 원본을 그대로 사용 (yaml 에서 원본 경로 참조)
  - train 만 구성별로 다르게 생성. 이미지는 하드링크(디스크 절약), 라벨은 신규 작성.
  - 클래스: 0=smoke, 1=fire (nc=2 유지 — 모든 셀 동일, val/test per-class AP 비교 가능)

구성:
  C1 (fire-only) : fire 포함 이미지(fire_only + both), smoke(0) 라벨 제거 → smoke 미학습
  C3 (균형 1:1)  : both + fire_only + smoke_only(다운샘플), fire:smoke 이미지 = 1:1, 라벨 그대로
  C4 (균형+NM)   : C3 + 정상배경(NM) 전체, 라벨 그대로 (배경은 빈 라벨)
"""
from __future__ import annotations
from pathlib import Path
import shutil

SRC = Path(r"C:\YangHyunHo\DFire\dfire\data")
OUT = Path(r"C:\YangHyunHo\DFire\compositions")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def categorize(split: str):
    """원본 train split 을 카테고리별 이미지 경로 리스트로 분류."""
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"
    buckets = {"fire_only": [], "smoke_only": [], "both": [], "background": []}
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        classes = set()
        n = 0
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                n += 1
                classes.add(int(line.split()[0]))
        if n == 0:
            buckets["background"].append((img, lbl))
        elif classes == {1}:
            buckets["fire_only"].append((img, lbl))
        elif classes == {0}:
            buckets["smoke_only"].append((img, lbl))
        elif classes == {0, 1}:
            buckets["both"].append((img, lbl))
    return buckets


def even_sample(items, k):
    """정렬된 리스트에서 균등 간격으로 k개 표본 (재현 가능, 결정적)."""
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def hardlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        import os
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_label_filtered(src_lbl: Path, dst_lbl: Path, keep_classes: set | None):
    """라벨 작성. keep_classes=None 이면 전체 복사, 아니면 해당 클래스 라인만."""
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    if not src_lbl.exists():
        dst_lbl.write_text("")  # 배경
        return
    if keep_classes is None:
        dst_lbl.write_text(src_lbl.read_text())
        return
    kept = []
    for line in src_lbl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if int(line.split()[0]) in keep_classes:
            kept.append(line)
    dst_lbl.write_text("\n".join(kept) + ("\n" if kept else ""))


def build_train(name: str, items: list, keep_classes: set | None):
    """items: [(img, lbl), ...] 를 OUT/name/train 에 생성."""
    img_out = OUT / name / "train" / "images"
    lbl_out = OUT / name / "train" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in items:
        hardlink(img, img_out / img.name)
        write_label_filtered(lbl, lbl_out / (img.stem + ".txt"), keep_classes)
    print(f"  [{name}] train 이미지 {len(items)}장 생성")


def write_yaml(name: str, train_subdir: str):
    """val/test 는 원본 DFire 참조. train 만 구성 폴더."""
    src_fwd = str(SRC).replace("\\", "/")
    train_fwd = str((OUT / train_subdir / "train" / "images")).replace("\\", "/")
    content = f"""# DFire ablation 구성: {name}
# 클래스 0=smoke, 1=fire / val·test 는 전체 DFire 고정
train: {train_fwd}
val: {src_fwd}/val/images
test: {src_fwd}/test/images

names: ['smoke', 'fire']
nc: 2
"""
    yaml_path = OUT / f"{name}.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"  yaml 작성: {yaml_path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1] train split 분류 중...")
    b = categorize("train")
    n_fire_only = len(b["fire_only"])
    n_smoke_only = len(b["smoke_only"])
    n_both = len(b["both"])
    n_bg = len(b["background"])
    print(f"    fire_only={n_fire_only} smoke_only={n_smoke_only} both={n_both} bg={n_bg}")

    # ── C1: fire 포함 이미지, smoke 라벨 제거 (keep class {1})
    print("[2] C1 (fire-only) 빌드...")
    c1_items = b["fire_only"] + b["both"]
    build_train("c1_fire_only", c1_items, keep_classes={1})

    # ── C3: 엄격한 1:1. fire이미지 = both + fire_only = n_both + n_fire_only
    #     smoke이미지도 동일 수가 되도록 smoke_only 를 다운샘플.
    #     both 는 fire·smoke 양쪽에 기여하므로 both 전체 포함.
    #     fire이미지수 = both + fire_only ; smoke이미지수 = both + smoke_sample
    #     1:1 → smoke_sample = fire_only
    print("[3] C3 (균형 1:1) 빌드...")
    smoke_sample = even_sample(b["smoke_only"], n_fire_only)
    c3_items = b["both"] + b["fire_only"] + smoke_sample
    n_fire_img = n_both + n_fire_only
    n_smoke_img = n_both + len(smoke_sample)
    print(f"    fire포함={n_fire_img} smoke포함={n_smoke_img} (목표 1:1)")
    build_train("c3_balanced", c3_items, keep_classes=None)

    # ── C4: C3 + NM 전체
    print("[4] C4 (균형+NM) 빌드...")
    c4_items = c3_items + b["background"]
    build_train("c4_balanced_nm", c4_items, keep_classes=None)

    # ── yaml 작성 (c4_yolo11s 는 c4 와 동일 데이터)
    print("[5] yaml 작성...")
    write_yaml("c1_fire_only", "c1_fire_only")
    write_yaml("c3_balanced", "c3_balanced")
    write_yaml("c4_balanced_nm", "c4_balanced_nm")
    write_yaml("c4_yolo11s", "c4_balanced_nm")  # 동일 데이터, 모델만 다름

    print("\n완료. 구성 요약:")
    print(f"  C1: {len(c1_items)}장 (smoke 미학습)")
    print(f"  C3: {len(c3_items)}장 (fire:smoke 이미지 1:1)")
    print(f"  C4: {len(c4_items)}장 (C3 + NM {n_bg})")


if __name__ == "__main__":
    main()
