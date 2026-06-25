"""정성 예시 그림 — C4(YOLO11n) best.pt 로 DFire test 추론, 박스+confidence 렌더.

공통 스타일(총괄세션): 제목 없음, detector 출력 그대로(class fire/smoke + conf).
출력 (04_figures/dfire_4cell/):
  fig_qualitative_success.png — 성공 2장 (화염 탐지 + 연기 탐지)
  fig_qualitative_failure.png — 실패/한계 1~2장 (정상배경 오탐 또는 미탐)
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("WANDB_MODE", "offline")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\YangHyunHo\DFire")
BEST = BASE / "runs" / "E_C4" / "weights" / "best.pt"
TEST_IMG = BASE / "dfire" / "data" / "test" / "images"
TEST_LBL = BASE / "dfire" / "data" / "test" / "labels"
OUT = BASE / "04_figures" / "dfire_4cell"


def gt_classes(stem):
    p = TEST_LBL / f"{stem}.txt"
    if not p.exists():
        return set()
    return {int(l.split()[0]) for l in p.read_text().splitlines() if l.strip()}


def categorize():
    fire_only, smoke_only, background = [], [], []
    for img in sorted(TEST_IMG.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        c = gt_classes(img.stem)
        if not c:
            background.append(img)
        elif c == {1}:
            fire_only.append(img)
        elif c == {0}:
            smoke_only.append(img)
    return fire_only, smoke_only, background


def render(model, img_path):
    """추론 후 박스 렌더된 RGB 이미지 + 탐지 (cls,conf) 리스트 반환. 실패 시 None."""
    try:
        r = model(str(img_path), verbose=False)[0]
        rgb = r.plot()[:, :, ::-1]  # BGR→RGB
        dets = [(int(b.cls), float(b.conf)) for b in r.boxes]
        return rgb, dets
    except Exception as e:
        return None


def pick(model, imgs, want_cls, min_conf=0.5, limit=200):
    """want_cls(0/1) 를 min_conf 이상으로 탐지한 첫 이미지."""
    for img in imgs[:limit]:
        out = render(model, img)
        if out is None:
            continue
        rgb, dets = out
        if any(c == want_cls and cf >= min_conf for c, cf in dets):
            return img, rgb, dets
    return None


def pick_fp(model, backgrounds, limit=300):
    """정상배경인데 박스를 친 첫 이미지(오탐/false alarm)."""
    for img in backgrounds[:limit]:
        out = render(model, img)
        if out is None:
            continue
        rgb, dets = out
        if dets:
            return img, rgb, dets
    return None


def save_panels(items, out, ncol):
    n = len(items)
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 5.0))
    if ncol == 1:
        axes = [axes]
    for ax, (rgb, _) in zip(axes, items):
        ax.imshow(rgb); ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[done] {out}")


def main():
    from ultralytics import YOLO
    model = YOLO(str(BEST))
    fire_only, smoke_only, background = categorize()
    print(f"test 카테고리: fire_only={len(fire_only)} smoke_only={len(smoke_only)} bg={len(background)}")

    succ = []
    f = pick(model, fire_only, want_cls=1, min_conf=0.55)
    s = pick(model, smoke_only, want_cls=0, min_conf=0.55)
    if f: print(f"성공(화염): {f[0].name} dets={f[2]}"); succ.append((f[1], f[2]))
    if s: print(f"성공(연기): {s[0].name} dets={s[2]}"); succ.append((s[1], s[2]))
    save_panels(succ, OUT / "fig_qualitative_success.png", ncol=2)

    fails = []
    fp = pick_fp(model, background)
    if fp:
        print(f"실패(정상배경 오탐): {fp[0].name} dets={fp[2]}"); fails.append((fp[1], fp[2]))
    # 미탐(FN): GT fire 있는데 탐지 0
    for img in fire_only[:300]:
        out = render(model, img)
        if out is None:
            continue
        rgb, dets = out
        if not dets:
            print(f"실패(미탐): {img.name}"); fails.append((rgb, dets)); break
    save_panels(fails, OUT / "fig_qualitative_failure.png", ncol=max(1, len(fails)))


if __name__ == "__main__":
    main()
