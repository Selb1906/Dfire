"""AIHub C1~C4 클래스 co-occurrence 카운트 (순수 카운팅, 학습 무관).

라벨(0=smoke, 1=fire) 기준 이미지별 분류: 화염만/연기만/화염+연기 동시/정상배경(NM).
목적: "AIHub는 화염·연기 동시 라벨 비율이 높아 D-Fire식 배타적 구성표 대신 텍스트로 설명" 근거.
출력: runs/aihub_cooccur.json + 콘솔 표.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE = Path(r"C:\YangHyunHo\DFire")
# (셀, 라벨 소스): dir=라벨폴더 직접 / list=이미지리스트→라벨매핑
CELLS = [
    ("C1", "dir", r"D:\AIHub_Fire\yolo_071751_c1\labels\train"),
    ("C2", "dir", r"D:\AIHub_Fire\yolo_071751_c2\labels\train"),
    ("C3", "list", str(BASE / "compositions" / "aihub_c3_train.txt")),
    ("C4", "dir", r"D:\AIHub_Fire\yolo_071751\labels\train"),
]


def classify(label_path: Path):
    """라벨 파일 → 'fire_only'|'smoke_only'|'both'|'nm'."""
    try:
        lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    except Exception:
        return "nm"
    if not lines:
        return "nm"
    cls = set(int(l.split()[0]) for l in lines)   # 0=smoke, 1=fire
    has_s, has_f = 0 in cls, 1 in cls
    if has_f and has_s:
        return "both"
    if has_f:
        return "fire_only"
    if has_s:
        return "smoke_only"
    return "nm"


def label_paths(kind, src):
    if kind == "dir":
        return [p for p in Path(src).iterdir() if p.suffix == ".txt"]
    # list: 이미지 경로 → 라벨 경로
    paths = []
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        lp = line.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
        lp = str(Path(lp).with_suffix(".txt"))
        paths.append(Path(lp))
    return paths


def main():
    results = {}
    print(f"{'셀':4} {'화염만':>10} {'연기만':>10} {'동시':>10} {'정상NM':>10} {'합계':>10} {'동시%':>7}")
    for cell, kind, src in CELLS:
        c = {"fire_only": 0, "smoke_only": 0, "both": 0, "nm": 0}
        for lp in label_paths(kind, src):
            c[classify(lp)] += 1
        total = sum(c.values())
        co_pct = round(c["both"] / total * 100, 1) if total else 0.0
        results[cell] = {**c, "total": total, "cooccur_pct": co_pct}
        print(f"{cell:4} {c['fire_only']:>10,} {c['smoke_only']:>10,} {c['both']:>10,} "
              f"{c['nm']:>10,} {total:>10,} {co_pct:>6.1f}%")
    (BASE / "runs" / "aihub_cooccur.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[저장] runs/aihub_cooccur.json")
    # 본문 인용용 한 줄
    c4 = results["C4"]
    print(f"[인용] C4 기준 전체 {c4['total']:,}장 중 화염+연기 동시 라벨 {c4['both']:,}장 = {c4['cooccur_pct']}%")


if __name__ == "__main__":
    main()
