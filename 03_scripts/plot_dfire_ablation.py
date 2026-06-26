"""DFire R8 ablation 그림 — runs/4cell_summary.json 수치만 사용 (GPU 불요).

공통 스타일(총괄세션 지침): 영문 텍스트만, 제목(title) 없음, 한국어 주석 없음.
생성 (04_figures/dfire_4cell/):
  fig_data_composition_dfire.png — 구성별 test mAP@0.5 + smoke AP (11n: C1·C2·C3·C4)
  fig_model_dfire.png            — 모델별 test mAP@0.5 (C4: 11n vs 11s)
  fig_smoke_ap_trend_dfire.png   — 구성별 test smoke AP 추이
수치 출처(SSOT): 02_data_ssot/TRAINING_LOG.md R8 (이 스크립트는 summary json 직접 파싱)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"   # 영문 전용
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
SUMMARY = BASE / "runs" / "4cell_summary.json"
MSTATS = BASE / "runs" / "multiseed_stats.json"   # 멀티시드 평균±표준편차(n=3)
OUT_DIR = BASE / "04_figures" / "dfire_4cell"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 구성 시리즈 (11n 고정), 영문 라벨
COMP_ORDER = [
    ("E_C1", "C1\nFlame only"),
    ("E_C2", "C2\nImbalanced 14:1"),
    ("E_C3", "C3\nBalanced 1:1"),
    ("E_C4", "C4\nBalanced+NM"),
]
PALETTE = {"C1": "#9aa0a6", "C2": "#c0392b", "C3": "#5b8def", "C4": "#1e7d32"}


def load() -> dict:
    return {r["name"]: r for r in json.loads(SUMMARY.read_text(encoding="utf-8"))}


def load_ms() -> dict:
    """멀티시드 통계(label → {mean, std}). 없으면 빈 dict."""
    if not MSTATS.exists():
        return {}
    return {r["label"]: r for r in json.loads(MSTATS.read_text(encoding="utf-8"))}


def _rows(by_name):
    rows = [(lab, by_name[n]["test"]) for n, lab in COMP_ORDER if n in by_name]
    labels = [lab for lab, _ in rows]
    return rows, labels


def plot_composition(by_name, ms) -> Path:
    rows, labels = _rows(by_name)
    maps, errs, smoke = [], [], []
    for lab, t in rows:
        key = lab.split("\n")[0]                 # C1..C4
        if key in ms:                            # C3/C4 → 멀티시드 평균±std
            maps.append(round(ms[key]["mean"], 3)); errs.append(ms[key]["std"])
        else:                                    # C1/C2 → 단일 시드(std 없음)
            maps.append(round(t["map50"], 3));   errs.append(0.0)
        smoke.append(round(t["smoke_ap50"], 3))
    colors = [PALETTE[l.split("\n")[0]] for l in labels]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = list(range(len(labels)))
    ax.bar(x, maps, color=colors, width=0.62, zorder=3, label="mAP@0.5 (2-class mean, n=3)")
    # 오차막대(±std) — 멀티시드 있는 셀(C3/C4)만
    ex = [(xi, m, e) for xi, m, e in zip(x, maps, errs) if e > 0]
    if ex:
        ax.errorbar([a for a, _, _ in ex], [b for _, b, _ in ex],
                    yerr=[c for _, _, c in ex], fmt="none", ecolor="black",
                    capsize=4, elinewidth=1.2, zorder=5)
    wbox = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0)
    for xi, v, e in zip(x, maps, errs):
        txt = f"{v:.3f}" + (f"\n±{e:.3f}" if e > 0 else "")
        ax.text(xi - 0.20, v + e + 0.010, txt, ha="center", va="bottom",
                fontsize=10, fontweight="bold", zorder=6, bbox=wbox)
    ax.plot(x, smoke, "o--", color="#7e4ec2", markersize=7, zorder=4, label="smoke AP")
    for xi, s in zip(x, smoke):
        ax.text(xi + 0.16, s, f"{s:.3f}", ha="left", va="center",
                fontsize=8.5, color="#7e4ec2", zorder=6, bbox=wbox)

    idx = {l.split("\n")[0]: (xi, m) for xi, (l, m) in enumerate(zip(labels, maps))}
    def arrow(a, b):
        if a in idx and b in idx:
            (xa, ma), (xb, mb) = idx[a], idx[b]
            ax.annotate("", xy=(xb, mb), xytext=(xa, ma),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
            ax.text((xa + xb) / 2, max(ma, mb) + 0.03,
                    f"+{(mb - ma) * 100:.1f}%p", ha="center", fontsize=9)
    arrow("C1", "C3")   # balance 총효과
    arrow("C3", "C4")   # NM

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP@0.5 / AP", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig_data_composition_dfire.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


def plot_model(ms) -> Path:
    pairs = [("YOLO11n", "C4"), ("YOLO11s", "C4_11s")]
    pairs = [(lab, ms[k]) for lab, k in pairs if k in ms]
    labels = [l for l, _ in pairs]
    maps = [round(r["mean"], 3) for _, r in pairs]
    errs = [r["std"] for _, r in pairs]

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    x = list(range(len(labels)))
    ax.bar(x, maps, color=["#1e7d32", "#1565c0"], width=0.5, zorder=3)
    ax.errorbar(x, maps, yerr=errs, fmt="none", ecolor="black", capsize=4,
                elinewidth=1.2, zorder=5)
    for xi, v, e in zip(x, maps, errs):
        ax.text(xi, v + e + 0.0015, f"{v:.3f}\n±{e:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    if len(maps) == 2:
        ax.annotate("", xy=(1, maps[1]), xytext=(0, maps[0]),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
        ax.text(0.5, max(maps) + max(errs) + 0.010, f"+{(maps[1] - maps[0]) * 100:.1f}%p",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_ylim(min(maps) - max(errs) - 0.025, max(maps) + max(errs) + 0.022)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    fig.tight_layout()
    out = OUT_DIR / "fig_model_dfire.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


def plot_smoke_trend(by_name) -> Path:
    rows, labels = _rows(by_name)
    smoke = [round(t["smoke_ap50"], 3) for _, t in rows]
    colors = [PALETTE[l.split("\n")[0]] for l in labels]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = list(range(len(labels)))
    ax.bar(x, smoke, color=colors, width=0.62, zorder=3)
    for xi, v in zip(x, smoke):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("smoke AP", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    fig.tight_layout()
    out = OUT_DIR / "fig_smoke_ap_trend_dfire.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    by = load()
    ms = load_ms()
    print(f"[plot] 멀티시드 평균 반영: {sorted(ms)} (C3/C4/C4_11s)")
    for f in (plot_composition(by, ms), plot_model(ms), plot_smoke_trend(by)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
