"""DFire R8 ablation 그림 — runs/4cell_summary.json 수치만 사용 (GPU 불요).

생성 (04_figures/dfire_4cell/):
  fig_dfire_composition.png — 구성별 test mAP@0.5 + smoke AP (모델 고정 11n: C1·C2·C3·C4)
  fig_dfire_model.png       — 모델별 test mAP@0.5 (데이터 고정 C4: 11n vs 11s)

summary 에 있는 셀만 자동 반영 → C2 학습 완료 후 재실행하면 C2 막대가 채워진다.
수치 출처(SSOT): 02_data_ssot/TRAINING_LOG.md R8 (이 스크립트는 summary json 직접 파싱)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
SUMMARY = BASE / "runs" / "4cell_summary.json"
OUT_DIR = BASE / "04_figures" / "dfire_4cell"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 구성 순서·라벨 (11n 고정 시리즈)
COMP_ORDER = [
    ("E_C1", "C1\n화염 단독"),
    ("E_C2", "C2\n불균형 14:1"),
    ("E_C3", "C3\n균형 1:1"),
    ("E_C4", "C4\n균형+정상"),
]


def load() -> dict:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {r["name"]: r for r in data}


def plot_composition(by_name: dict) -> Path:
    rows = [(lab, by_name[n]["test"]) for n, lab in COMP_ORDER if n in by_name]
    labels = [lab for lab, _ in rows]
    maps = [t["map50"] for _, t in rows]
    smoke = [t["smoke_ap50"] for _, t in rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = range(len(labels))
    palette = {"C1": "#9aa0a6", "C2": "#c0392b", "C3": "#5b8def", "C4": "#1e7d32"}
    colors = [palette[lab.split("\n")[0]] for lab in labels]
    ax.bar(x, maps, color=colors, width=0.62, zorder=3, label="mAP@0.5 (2클래스 평균)")

    for xi, v in zip(x, maps):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    # smoke AP 점선
    ax.plot(list(x), smoke, "o--", color="#7e4ec2", markersize=7, zorder=4,
            label="smoke AP")
    for xi, s in zip(x, smoke):
        ax.text(xi, s + 0.012, f"{s:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#7e4ec2")

    # 효과 화살표 (인접 셀 존재 시)
    idx = {lab.split("\n")[0]: (xi, m) for xi, (lab, m) in enumerate(zip(labels, maps))}
    def arrow(a, b, text):
        if a in idx and b in idx:
            (xa, ma), (xb, mb) = idx[a], idx[b]
            ax.annotate("", xy=(xb, mb), xytext=(xa, ma),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
            ax.text((xa + xb) / 2, max(ma, mb) + 0.03, text, ha="center", fontsize=8.5)
    arrow("C1", "C3", f"+{(idx.get('C3',(0,0))[1]-idx.get('C1',(0,0))[1])*100:.1f}%p\n(클래스 균형)")
    arrow("C3", "C4", f"+{(idx.get('C4',(0,0))[1]-idx.get('C3',(0,0))[1])*100:.1f}%p\n(정상배경)")

    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP@0.5 / AP", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("데이터 구성에 따른 화재·연기 탐지 성능 (DFire, YOLO11n 고정)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig_data_composition_dfire.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


def plot_model(by_name: dict) -> Path:
    rows = [("YOLO11n", "E_C4"), ("YOLO11s", "E_C4_11s")]
    rows = [(lab, by_name[n]["test"]["map50"]) for lab, n in rows if n in by_name]
    labels = [l for l, _ in rows]; maps = [m for _, m in rows]

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    x = range(len(labels))
    ax.bar(x, maps, color=["#1e7d32", "#1565c0"], width=0.5, zorder=3)
    for xi, v in zip(x, maps):
        ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    if len(maps) == 2:
        ax.annotate("", xy=(1, maps[1]), xytext=(0, maps[0]),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
        ax.text(0.5, max(maps) + 0.006, f"+{(maps[1]-maps[0])*100:.1f}%p",
                ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    lo = min(maps) - 0.03; ax.set_ylim(lo, max(maps) + 0.02)
    ax.set_title("모델 용량에 따른 성능 (DFire, 데이터 고정: C4 균형+정상)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    fig.tight_layout()
    out = OUT_DIR / "fig_model_dfire.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    by_name = load()
    present = [n for n, _ in COMP_ORDER if n in by_name]
    print(f"[plot] 구성 시리즈 반영 셀: {present}"
          + ("" if "E_C2" in by_name else "  (C2 미완료 → 완료 후 재실행하면 채워짐)"))
    a = plot_composition(by_name)
    b = plot_model(by_name)
    print(f"[plot] 생성 완료:\n  {a}\n  {b}")


if __name__ == "__main__":
    main()
