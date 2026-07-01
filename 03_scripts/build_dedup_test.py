"""중복제거 test 서브셋 구축 — train과 근접중복(dHash 해밍거리 ≤ THRESH)인 test 이미지 제외.

발견: DFire Kaggle 분할에 train↔test 근접중복 누수(거리=0이 22%, ≤5가 51%). 픽셀 검증 완료(MSE~0).
보수적 기준(거리 ≤5) 제외 → 누수 없는 정직한 평가셋. 기존 val/test 원본은 그대로 두고 별도 폴더 생성.
출력: dfire_test_dedup/{images,labels} (하드링크) + compositions/dfire_dedup.yaml + 제외목록.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(r"C:\YangHyunHo\DFire\dfire\data")
OUT = Path(r"C:\YangHyunHo\DFire\dfire_test_dedup")
YAML = Path(r"C:\YangHyunHo\DFire\compositions\dfire_dedup.yaml")
IMG_EXT = {".jpg", ".jpeg", ".png"}
THRESH = 5   # 이 거리 이하면 근접중복으로 제외 (보수적)


def dhash64(path):
    try:
        a = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten(); v = 0
        for b in bits:
            v = (v << 1) | int(b)
        return np.uint64(v)
    except Exception:
        return None


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def paths(sp):
    return [p for p in sorted((ROOT / sp / "images").iterdir()) if p.suffix.lower() in IMG_EXT]


def hardlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil; shutil.copy2(src, dst)


def main():
    trp, tep = paths("train"), paths("test")
    print(f"dHash 계산: train {len(trp)}, test {len(tep)}")
    trh = np.array([dhash64(p) for p in trp], dtype=np.uint64)

    img_out = OUT / "images"; lbl_out = OUT / "labels"
    kept, excluded = [], []
    for tp in tep:
        h = dhash64(tp)
        if h is None:
            continue
        mind = int(popcount64(trh ^ h).min())
        if mind <= THRESH:
            excluded.append(tp.name)
        else:
            kept.append(tp)
            hardlink(tp, img_out / tp.name)
            lb = ROOT / "test" / "labels" / (tp.stem + ".txt")
            if lb.exists():
                hardlink(lb, lbl_out / (tp.stem + ".txt"))

    (OUT / "excluded_list.txt").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "excluded_list.txt").write_text("\n".join(excluded), encoding="utf-8")
    src_fwd = str(ROOT).replace("\\", "/")
    YAML.write_text(
        f"# DFire dedup — train 근접중복(dHash ≤{THRESH}) 제외한 test. val은 원본.\n"
        f"# 누수 검증용. 원본 test={len(tep)}, 제외={len(excluded)}, 유지={len(kept)}\n"
        f"train: {src_fwd}/train/images\n"
        f"val: {src_fwd}/val/images\n"
        f"test: {OUT.as_posix()}/images\n\nnames: ['smoke', 'fire']\nnc: 2\n",
        encoding="utf-8")
    print(f"[done] 원본 test {len(tep)} → 제외 {len(excluded)} (거리≤{THRESH}) → dedup test {len(kept)}")
    print(f"  yaml: {YAML}")


if __name__ == "__main__":
    main()
