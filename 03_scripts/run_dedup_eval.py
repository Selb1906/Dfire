"""중복제거 재평가 — 주요 모델을 dedup test(2,098, 근접중복 제외)로 재평가.

목적: train↔test 근접중복 누수의 영향을 통제 실험으로 확인. dedup 수치가 full-test와 비슷하면
'누수 영향 미미(강건성 근거)', 크게 낮으면 지금 파악해 정직히 보고.
대상: C1~C4(ablation) + D_full(최종). 기존 best.pt로 평가만(재학습 없음).
"""
from __future__ import annotations
import json, os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
DEDUP_YAML = str(BASE / "compositions" / "dfire_dedup.yaml")

# (표시이름, run이름, imgsz) — best.pt 없으면 자동 건너뜀
TARGETS = [
    ("C1", "E_C1", 640), ("C2", "E_C2", 640), ("C3", "E_C3", 640), ("C4", "E_C4", 640),
    ("Dfull_11s", "Dfull_11s", 640), ("Dfull_11m", "Dfull_11m", 640), ("Dfull_11l", "Dfull_11l", 640),
]
# full-test 기준값(비교용): 4cell_summary + dfull_summary 에서 읽음


def metrics_dict(res) -> dict:
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None}


def full_test_map(run):
    for f in ["4cell_summary.json", "dfull_summary.json"]:
        p = Path(PROJECT) / f
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8")):
                if r.get("name") == run and "test" in r:
                    return r["test"]["map50"]
    return None


def main():
    from ultralytics import YOLO
    out = Path(PROJECT) / "dedup_eval_summary.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["run"] for r in results}
    for disp, run, imgsz in TARGETS:
        if run in done:
            print(f"[{disp}] 집계됨 — 건너뜀"); continue
        best = Path(PROJECT) / run / "weights" / "best.pt"
        if not best.exists():
            print(f"[{disp}] best.pt 없음(미완료?) — 건너뜀"); continue
        res = YOLO(str(best)).val(data=DEDUP_YAML, split="test", imgsz=imgsz, device="0",
                                  project=PROJECT, name=f"{run}_dedup", exist_ok=True,
                                  plots=False, verbose=False)
        m = metrics_dict(res)
        ft = full_test_map(run)
        rec = {"disp": disp, "run": run, "dedup_test": m,
               "full_test_map50": ft,
               "delta_map50": (round(m["map50"] - ft, 4) if ft is not None else None)}
        results.append(rec)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        d = rec["delta_map50"]
        print(f"[{disp}] dedup mAP50={m['map50']:.4f}  (full={ft})  Δ={d}")

    print("\n===== dedup vs full-test (mAP@0.5) =====")
    print(f"{'모델':<10} {'full':>7} {'dedup':>7} {'Δ':>8}")
    for r in results:
        ft = r["full_test_map50"]; dd = r["dedup_test"]["map50"]
        print(f"{r['disp']:<10} {(ft if ft else 0):>7.4f} {dd:>7.4f} {(r['delta_map50'] or 0):>+8.4f}")


if __name__ == "__main__":
    main()
