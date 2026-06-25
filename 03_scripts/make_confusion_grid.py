"""DFire C1~C4 정규화 혼동행렬 2x2 그리드 (데이터 재계산, 단일 공유 컬러바, 제목 없음).

각 best.pt 로 DFire test 재평가 → confusion matrix 재산출 → 커스텀 히트맵.
공통 스타일(총괄세션): 영문만, 개별 title 없음, 패널 소형 레이블 (a)~(d)만, 컬러바 1개 우측 공유.
출력: 04_figures/dfire_4cell/fig_confusion_matrix_comp_dfire.png (300dpi)
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
OUT = BASE / "04_figures" / "dfire_4cell" / "fig_confusion_matrix_comp_dfire.png"
CLASSES = ["smoke", "fire", "background"]
# (run 이름, data yaml, 패널 레이블) — 배치: C1 좌상 / C2 우상 / C3 좌하 / C4 우하
CELLS = [
    ("E_C1", "compositions/c1_fire_only.yaml",   "(a) C1: Flame only"),
    ("E_C2", "compositions/c2_imbalanced.yaml",  "(b) C2: Imbalanced 14:1"),
    ("E_C3", "compositions/c3_balanced.yaml",    "(c) C3: Balanced 1:1"),
    ("E_C4", "compositions/c4_balanced_nm.yaml", "(d) C4: Balanced+NM"),
]


def confusion(run, data):
    from ultralytics import YOLO
    model = YOLO(str(BASE / "runs" / run / "weights" / "best.pt"))
    # plots=True 여야 confusion matrix 가 누적·반환됨 (plots=False면 빈 행렬)
    res = model.val(data=str(BASE / data), split="test", device="0",
                    verbose=False, plots=True)
    m = np.array(res.confusion_matrix.matrix, dtype=float)   # (3,3): 행=pred, 열=true
    assert m.sum() > 0, f"{run}: 빈 confusion matrix (접근경로 점검 필요)"
    col = m.sum(0, keepdims=True); col[col == 0] = 1
    return m / col                                            # 열(true) 기준 정규화


def main():
    mats = [(confusion(r, d), lab) for r, d, lab in CELLS]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    im = None
    for ax, (mat, lab) in zip(axes.flat, mats):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(lab, fontsize=12, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, fontsize=9); ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel("True", fontsize=10); ax.set_ylabel("Predicted", fontsize=10)
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if v > 0.5 else "black")
    fig.tight_layout(rect=(0, 0, 0.91, 1))
    cax = fig.add_axes((0.93, 0.15, 0.02, 0.7))
    fig.colorbar(im, cax=cax)
    fig.savefig(OUT, dpi=300); plt.close(fig)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
