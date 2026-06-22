"""Ablation 막대그래프 생성 — 가중치 불요, TRAINING_LOG.md 수치만 사용.

생성 그림 (04_figures/):
  fig_data_composition.png  — 실험 A: 데이터 구성별 mAP@0.5 + smoke AP
  fig_model_capacity.png    — 실험 B: 모델 용량별 mAP@0.5

수치 출처(SSOT): 02_data_ssot/TRAINING_LOG.md
  C1=R2, C2=R3, C3=R4, C4=E01 / 11n=E01, 11s=E02, 11m=E03, 11l=E04(발산)

사용법:
  python 03_scripts/plot_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 폰트 (Windows)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent.parent / "04_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 실험 A: 데이터 구성 (모델 고정 YOLO11n) ──────────────────────
COMP_LABELS = ["C1\n화염 단독", "C2\n불균형 14:1", "C3\n균형 1:1", "C4\n균형+정상"]
COMP_MAP = [0.777, 0.761, 0.836, 0.911]
COMP_SMOKE = [None, 0.684, 0.792, None]  # 측정 가능한 구성만


def plot_data_composition() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = range(len(COMP_LABELS))
    colors = ["#9aa0a6", "#c0392b", "#5b8def", "#1e7d32"]
    bars = ax.bar(x, COMP_MAP, color=colors, width=0.62, zorder=3)

    for xi, v in zip(x, COMP_MAP):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    # smoke AP 점 표시 (측정된 구성만)
    sx = [xi for xi, s in zip(x, COMP_SMOKE) if s is not None]
    sy = [s for s in COMP_SMOKE if s is not None]
    ax.plot(sx, sy, "o--", color="#7e4ec2", markersize=7, zorder=4,
            label="smoke AP")
    for xi, s in zip(sx, sy):
        ax.text(xi, s - 0.018, f"{s:.3f}", ha="center", va="top",
                fontsize=9, color="#7e4ec2")

    # 개선폭 화살표 주석
    ax.annotate("", xy=(2, 0.836), xytext=(1, 0.761),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
    ax.text(1.5, 0.806, "+7.5%p\n(클래스 균형)", ha="center", fontsize=8.5)
    ax.annotate("", xy=(3, 0.911), xytext=(2, 0.836),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
    ax.text(2.5, 0.882, "+7.5%p\n(정상배경)", ha="center", fontsize=8.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(COMP_LABELS, fontsize=10)
    ax.set_ylabel("mAP@0.5 / AP", fontsize=11)
    ax.set_ylim(0.65, 0.95)
    ax.set_title("데이터 구성에 따른 화재·연기 탐지 성능 (YOLO11n 고정)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = OUT_DIR / "fig_data_composition.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ── 실험 B: 모델 용량 (데이터 고정 C4) ──────────────────────────
MODEL_LABELS = ["YOLO11n", "YOLO11s", "YOLO11m", "YOLO11l"]
MODEL_MAP = [0.911, 0.918, 0.914, None]  # 11l = 발산 실패


def plot_model_capacity() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = range(len(MODEL_LABELS))
    vals = [v if v is not None else 0 for v in MODEL_MAP]
    colors = ["#1e7d32", "#1565c0", "#6a1b9a", "#bdbdbd"]
    ax.bar(x, vals, color=colors, width=0.58, zorder=3)

    for xi, v in zip(x, MODEL_MAP):
        if v is None:
            ax.text(xi, 0.902, "발산 실패\n(수렴 X)", ha="center", va="bottom",
                    fontsize=9.5, color="#b00020", fontweight="bold")
        else:
            ax.text(xi, v + 0.0008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    ax.annotate("", xy=(1, 0.918), xytext=(0, 0.911),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
    ax.text(0.5, 0.9205, "+0.7%p", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_ylim(0.900, 0.925)
    ax.set_title("모델 용량에 따른 성능 (데이터 고정: C4 균형+정상 175K)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    fig.tight_layout()

    out = OUT_DIR / "fig_model_capacity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = plot_data_composition()
    b = plot_model_capacity()
    print(f"[plot] 생성 완료:\n  {a}\n  {b}")


if __name__ == "__main__":
    main()
