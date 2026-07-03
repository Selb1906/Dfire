"""§5.1 헤드라인 — AIHub C1→C4 구성 ablation. runs/aihub_c3_summary.json 수치만 (GPU 불요).

지침: 영문, 제목 없음. 지표 3개(mAP@0.5, smoke AP, fire AP)만(P/R/mAP50:95 제외).
4구성 × 3지표 = 12막대. C3=summary['val'], C1/C2/C4=summary['sanity_reeval'](fresh, 없으면 R9 상수).
AIHub 클래스 0=smoke,1=fire. 출력: 04_figures/fig_aihub_composition_headline.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "runs" / "aihub_c3_summary.json"
OUT = BASE / "04_figures" / "fig_aihub_composition_headline.png"

# R9 fallback (map50, smoke_ap50, fire_ap50) — sanity 재평가 없을 때만
R9 = {"C1": (0.463, 0.000, 0.927), "C2": (0.770, 0.616, 0.925), "C4": (0.913, 0.896, 0.930)}
CFG = [("C1", "C1\nFlame only"), ("C2", "C2\nImbalanced 14:1"),
       ("C3", "C3\nBalanced 1:1"), ("C4", "C4\nBalanced+NM")]
METRICS = [("map50", "mAP@0.5", "#1565c0"), ("smoke_ap50", "smoke AP", "#ef6c00"),
           ("fire_ap50", "fire AP", "#2e7d32")]


def cell_vals(c3):
    sr = c3.get("sanity_reeval", {})
    v = {}
    for key in ("C1", "C2", "C4"):
        if key in sr:
            v[key] = (sr[key]["map50"], sr[key]["smoke_ap50"], sr[key]["fire_ap50"])
        else:
            v[key] = R9[key]
    cv = c3["val"]
    v["C3"] = (cv["map50"], cv["smoke_ap50"], cv["fire_ap50"])
    return v


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c3 = json.loads(SRC.read_text(encoding="utf-8"))
    v = cell_vals(c3)

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    x = list(range(len(CFG))); w = 0.26
    offs = [-w, 0, w]
    for mi, (mk, mlab, col) in enumerate(METRICS):
        vals = [v[key][mi] for key, _ in CFG]
        bars = ax.bar([xi + offs[mi] for xi in x], vals, width=w, color=col, zorder=3, label=mlab)
        for r, val in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, val + 0.010, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=6, rotation=90)

    # mAP@0.5 상승 화살표 (C1→C4 헤드라인)
    maps = [v[key][0] for key, _ in CFG]
    for a in range(len(x) - 1):
        ax.annotate("", xy=(x[a + 1] - w, maps[a + 1]), xytext=(x[a] - w, maps[a]),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.1, alpha=0.7))
    ax.text(0.5 * (x[0] + x[-1]), max(maps) + 0.085,
            f"mAP@0.5: {maps[0]:.3f} → {maps[-1]:.3f}  (+{(maps[-1]-maps[0])*100:.1f}%p)",
            ha="center", fontsize=9.5, fontweight="bold", color="#1565c0")

    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in CFG], fontsize=9.5)
    ax.set_ylabel("mAP@0.5 / AP", fontsize=11)
    ax.set_ylim(0.0, 1.14)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9, ncol=3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300); plt.close(fig)
    print(f"[plot] {OUT}")
    for key, _ in CFG:
        print(f"  {key}: mAP={v[key][0]:.3f} smoke={v[key][1]:.3f} fire={v[key][2]:.3f}")


if __name__ == "__main__":
    main()
