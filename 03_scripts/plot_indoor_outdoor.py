"""Figure 1 (§5.5) — AIHub C4 실내 vs 실외 (raw). runs/aihub_val_inout.json 수치만 사용 (GPU 불요).

지침: 영문 텍스트, 제목 없음. 지표 3개(mAP@0.5, fire AP, smoke AP)만(P/R/mAP50:95 제외).
raw만 사용(실내 0.946/0.824 등) — 클래스통제(0.958/0.821)는 그래프에 넣지 않음(표·본문 전용).
포인트: fire AP 격차(0.980 vs 0.765) ≫ smoke AP 격차(0.911 vs 0.883)가 한눈에.
출력: 04_figures/fig_indoor_outdoor_aihub.png
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
SRC = BASE / "runs" / "aihub_val_inout.json"
OUT = BASE / "04_figures" / "fig_indoor_outdoor_aihub.png"

METRICS = [("map50", "mAP@0.5"), ("fire_ap50", "fire AP"), ("smoke_ap50", "smoke AP")]
C_IN, C_OUT = "#1565c0", "#ef6c00"   # Indoor 파랑 / Outdoor 주황


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = json.loads(SRC.read_text(encoding="utf-8"))
    ind = [d["in"][k] for k, _ in METRICS]
    out = [d["out"][k] for k, _ in METRICS]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = list(range(len(METRICS))); w = 0.36
    b1 = ax.bar([xi - w / 2 for xi in x], ind, width=w, color=C_IN, zorder=3, label="Indoor")
    b2 = ax.bar([xi + w / 2 for xi in x], out, width=w, color=C_OUT, zorder=3, label="Outdoor")
    for bars, vals in ((b1, ind), (b2, out)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=6)
    # 지표별 Δ(실내-실외) 주석 — fire 격차 >> smoke 격차 강조
    for xi, (vi, vo) in enumerate(zip(ind, out)):
        gap = vi - vo
        y = max(vi, vo) + 0.075
        ax.annotate("", xy=(xi + w / 2, max(vi, vo) + 0.055), xytext=(xi - w / 2, max(vi, vo) + 0.055),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.1))
        ax.text(xi, y, f"Δ {gap:.3f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#b00020" if gap > 0.1 else "#444", zorder=6)

    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in METRICS], fontsize=11)
    ax.set_ylabel("AP", fontsize=11)
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300); plt.close(fig)
    print(f"[plot] {OUT}")
    print(f"  Indoor  {dict((l, round(v,3)) for (_,l),v in zip(METRICS, ind))}")
    print(f"  Outdoor {dict((l, round(v,3)) for (_,l),v in zip(METRICS, out))}")
    print(f"  Δ fire={ind[1]-out[1]:.3f}  Δ smoke={ind[2]-out[2]:.3f}  (fire 격차가 훨씬 큼)")


if __name__ == "__main__":
    main()
