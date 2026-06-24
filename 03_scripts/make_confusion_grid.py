"""DFire C1~C4 정규화 혼동행렬을 2x2 그리드로 합쳐 논문 그림 생성 (GPU 불요).

입력: 04_figures/dfire_4cell/C{1..4}_test_confusion_matrix_norm.png
출력: 04_figures/dfire_4cell/fig_confusion_matrix_comp_dfire.png (300dpi)
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DIR = Path(__file__).resolve().parent.parent / "04_figures" / "dfire_4cell"
CELLS = [("C1", "C1 화염 단독"), ("C2", "C2 불균형 14:1"),
         ("C3", "C3 균형 1:1"), ("C4", "C4 균형+정상")]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (cell, title) in zip(axes.flat, CELLS):
        p = DIR / f"{cell}_test_confusion_matrix_norm.png"
        if not p.exists():
            ax.text(0.5, 0.5, f"{cell}\n(없음)", ha="center", va="center")
            ax.axis("off"); continue
        ax.imshow(mpimg.imread(p)); ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold")
    fig.suptitle("DFire 데이터 구성별 정규화 혼동행렬 (test, YOLO11n)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = DIR / "fig_confusion_matrix_comp_dfire.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
