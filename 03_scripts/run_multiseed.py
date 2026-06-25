"""멀티시드 실험 — 5개 구성 × 추가 seed 2회(원래 seed=0 포함 3회 반복) → test mAP@0.5 평균±표준편차.

기록: E11~E20 (각 구성 seed1/seed2). 학습 설정은 기존과 100% 동일, seed만 변경.
원래 seed=0 결과는 기존 summary(4cell/volcontrol)에서 읽어 3회 통계에 합산.
"""
from __future__ import annotations
import json, os, time, statistics
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DEVICE, EPOCHS, IMGSZ = "0", 100, 640
SEEDS = [1, 2]

# (label, yaml, model, batch, 원래 seed0 출처(summary파일, run이름))
CONFIGS = [
    ("C3",     "c3_balanced.yaml",    "yolo11n.pt", 64, "4cell_summary.json",      "E_C3"),
    ("C4",     "c4_balanced_nm.yaml", "yolo11n.pt", 64, "4cell_summary.json",      "E_C4"),
    ("C4_11s", "c4_balanced_nm.yaml", "yolo11s.pt", 48, "4cell_summary.json",      "E_C4_11s"),
    ("C3_vol", "c3_vol.yaml",         "yolo11n.pt", 64, "volcontrol_summary.json", "E09_C3_vol"),
    ("C4_eq",  "c4_eq.yaml",          "yolo11n.pt", 64, "volcontrol_summary.json", "E10_C4_eq"),
]


def test_map50(name, yaml, model_pt, batch, seed):
    from ultralytics import YOLO
    best = Path(PROJECT) / name / "weights" / "best.pt"
    if not best.exists():
        YOLO(model_pt).train(
            data=str(BASE / "compositions" / yaml), epochs=EPOCHS, imgsz=IMGSZ,
            batch=batch, device=DEVICE, project=PROJECT, name=name, exist_ok=True,
            save_period=0, patience=30, optimizer="AdamW", lr0=0.001, lrf=0.01,
            cos_lr=True, warmup_epochs=3, cache=False, seed=seed,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=False)
    res = YOLO(str(best)).val(data=str(BASE / "compositions" / yaml), imgsz=IMGSZ,
                              device=DEVICE, split="test", verbose=False, plots=False)
    return float(res.box.map50)


def orig_seed0(summary_file, run_name):
    data = json.loads((Path(PROJECT) / summary_file).read_text(encoding="utf-8"))
    return {r["name"]: r for r in data}[run_name]["test"]["map50"]


def main():
    out = Path(PROJECT) / "multiseed_summary.json"
    runs = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["label"], r["seed"]) for r in runs}

    for label, yaml, model_pt, batch, _, _ in CONFIGS:
        for seed in SEEDS:
            if (label, seed) in done:
                print(f"[{label} s{seed}] 완료됨 — 건너뜀"); continue
            name = f"{label}_s{seed}"
            t0 = time.time()
            m = test_map50(name, yaml, model_pt, batch, seed)
            runs.append({"label": label, "seed": seed, "name": name,
                         "test_map50": round(m, 4), "elapsed_sec": round(time.time() - t0, 1)})
            out.write_text(json.dumps(runs, indent=2, ensure_ascii=False))
            print(f"[{label} s{seed}] test mAP50={m:.4f} ({(time.time()-t0)/3600:.2f}h)")

    # 통계 집계 (seed0 + seed1 + seed2)
    print("\n===== 멀티시드 통계 (test mAP@0.5, 3회) =====")
    print(f"{'구성':<8} {'seed0':>7} {'seed1':>7} {'seed2':>7} {'평균':>8} {'표준편차':>9}")
    seedmap = {(r["label"], r["seed"]): r["test_map50"] for r in runs}
    table = []
    for label, _, _, _, sf, rn in CONFIGS:
        s0 = round(orig_seed0(sf, rn), 4)
        s1 = seedmap.get((label, 1)); s2 = seedmap.get((label, 2))
        vals = [v for v in [s0, s1, s2] if v is not None]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        table.append({"label": label, "seed0": s0, "seed1": s1, "seed2": s2,
                      "mean": round(mean, 4), "std": round(sd, 4), "n": len(vals)})
        print(f"{label:<8} {s0:>7.4f} {str(s1):>7} {str(s2):>7} {mean:>8.4f} {sd:>9.4f}")
    (Path(PROJECT) / "multiseed_stats.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False))
    print(f"\n집계: {out}\n통계: {Path(PROJECT)/'multiseed_stats.json'}")


if __name__ == "__main__":
    main()
