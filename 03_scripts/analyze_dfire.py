"""DFire 라벨 분포 분석 — C1/C3/C4 구성 설계용.

클래스: 0=smoke, 1=fire  (data.yaml 기준)
각 split별로 이미지를 카테고리로 분류:
  - fire_only  : fire 박스만 있음 (smoke 없음)
  - smoke_only : smoke 박스만 있음 (fire 없음)
  - both       : fire + smoke 둘 다
  - background : 라벨 파일이 비어있음 (정상배경 NM)
  - missing    : 라벨 파일 자체가 없음
"""
from __future__ import annotations
from pathlib import Path
import collections

ROOT = Path(r"C:\YangHyunHo\DFire\dfire\data")
SPLITS = ["train", "val", "test"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def analyze_split(split: str) -> dict:
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    cats = collections.Counter()
    inst = collections.Counter()  # 클래스별 박스 인스턴스 수
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    for img in imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            cats["missing"] += 1
            continue
        classes = set()
        n_lines = 0
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            cid = int(line.split()[0])
            classes.add(cid)
            inst[cid] += 1
        if n_lines == 0:
            cats["background"] += 1
        elif classes == {1}:
            cats["fire_only"] += 1
        elif classes == {0}:
            cats["smoke_only"] += 1
        elif classes == {0, 1}:
            cats["both"] += 1
        else:
            cats[f"other_{sorted(classes)}"] += 1
    return {"total": len(imgs), "cats": cats, "inst": inst}


def main():
    print(f"{'split':<8} {'total':>7} {'fire_only':>10} {'smoke_only':>11} {'both':>7} {'bg(NM)':>8} {'missing':>8}")
    print("-" * 70)
    grand = collections.Counter()
    grand_inst = collections.Counter()
    for sp in SPLITS:
        r = analyze_split(sp)
        c = r["cats"]
        print(f"{sp:<8} {r['total']:>7} {c.get('fire_only',0):>10} "
              f"{c.get('smoke_only',0):>11} {c.get('both',0):>7} "
              f"{c.get('background',0):>8} {c.get('missing',0):>8}")
        grand.update(c)
        grand.update({"total": r["total"]})
        grand_inst.update(r["inst"])
    print("-" * 70)
    print(f"{'TOTAL':<8} {grand['total']:>7} {grand.get('fire_only',0):>10} "
          f"{grand.get('smoke_only',0):>11} {grand.get('both',0):>7} "
          f"{grand.get('background',0):>8} {grand.get('missing',0):>8}")
    print()
    print("인스턴스(박스) 수:  smoke(0) =", grand_inst.get(0, 0),
          " fire(1) =", grand_inst.get(1, 0))
    # 기타 카테고리 표시
    for k, v in grand.items():
        if k.startswith("other"):
            print("  주의:", k, "=", v)


if __name__ == "__main__":
    main()
