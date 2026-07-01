"""AIHub 071751 메타데이터 감사 — JSON attributes 필드·값 분포 + 폴더(장소) 구조.

목적: 결합학습을 "AIHub의 어떤 부분(장소/실내외/화재종류 등)을 넣으면 D-Fire test가 어떻게 변하나"로
설계하기 위해, 어떤 메타데이터가 있는지 파악. clip 단위 1개 JSON 샘플(프레임은 clip 속성 공유).
"""
from __future__ import annotations
from pathlib import Path
import json
from collections import Counter, defaultdict

ROOT = Path(r"D:\AIHub_Fire\extracted\Training_labels")
BRANCHES = ["화재 현상", "화재현상"]


def find_branch(root):
    for b in BRANCHES:
        if (root / b).exists():
            return root / b
    return None


def main():
    branch = find_branch(ROOT)
    print(f"브랜치: {branch}")
    # clip당 첫 JSON 1개 샘플
    seen_clip = set()
    samples = []
    for jp in branch.rglob("JSON/*.json"):
        clip = jp.parent.parent.name
        if clip in seen_clip:
            continue
        seen_clip.add(clip)
        samples.append(jp)
    print(f"clip 수(=샘플): {len(samples)}\n")

    attr_fields = Counter()
    dists = defaultdict(Counter)     # 필드 → 값 분포
    place_by_class = defaultdict(Counter)   # 폴더 기반 class → 장소
    KEYS = ["class", "inout", "place", "fire_reason", "fire_level",
            "condition", "source", "device", "angle", "date"]
    for jp in samples:
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception:
            continue
        at = d.get("attributes", {}) or {}
        for k in at:
            attr_fields[k] += 1
        for k in KEYS:
            v = at.get(k)
            if k == "date" and isinstance(v, str):
                v = v[:4]  # 연도만
            dists[k][str(v)] += 1
        # 폴더 경로: .../{class폴더}/{장소폴더}/{clip}/JSON/x.json
        parts = jp.parts
        try:
            i = parts.index("이미지")
            cls_folder, place_folder = parts[i + 1], parts[i + 2]
            place_by_class[cls_folder][place_folder] += 1
        except (ValueError, IndexError):
            pass

    print("=== attributes 필드 존재율 (clip 기준) ===")
    for k, c in attr_fields.most_common():
        print(f"  {k}: {c}/{len(samples)}")

    print("\n=== 주요 필드 값 분포 (clip 기준) ===")
    for k in KEYS:
        vals = dists[k].most_common(12)
        print(f"  [{k}] 고유값 {len(dists[k])}종: {vals}")

    print("\n=== 폴더 구조: class → 장소 (clip 수) ===")
    for cls, places in place_by_class.items():
        print(f"  {cls}: {dict(places.most_common())}")


if __name__ == "__main__":
    main()
