"""X — 크로스평가: D-Fire 학습 모델(D_full) → AIHub val 평가 (도메인 매트릭스 반대 방향).

A-1(AIHub→DFire=0.262)의 반대: DFire→AIHub. 두 방향 모두 급락하면 '양방향 도메인 갭' 확정.
재학습 없음. D_full 11m(헤드라인)·11s(전이 비교 동급) 두 개 평가. runs/domain_matrix.json 저장.
"""
from __future__ import annotations
import json, os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
AIHUB_VAL = str(BASE / "compositions" / "aihub_val.yaml")
IMGSZ, DEVICE = 640, "0"
MODELS = [("Dfull_11m", "Dfull_11m/weights/best.pt"),
          ("Dfull_11s", "Dfull_11s/weights/best.pt")]


def m(res):
    ap50 = [float(v) for v in res.box.ap50]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None}


def main():
    from ultralytics import YOLO
    rows = []
    for name, rel in MODELS:
        best = BASE / "runs" / rel
        if not best.exists():
            print(f"[스킵] {name} 없음: {best}"); continue
        r = YOLO(str(best)).val(data=AIHUB_VAL, imgsz=IMGSZ, device=DEVICE, split="val",
                                project=PROJECT, name=f"{name}_on_AIHubval", exist_ok=True,
                                plots=False, verbose=False)
        d = m(r); d["model"] = name
        rows.append(d)
        print(f"[{name}] → AIHub val mAP50={d['map50']:.4f} fireAP={d['fire_ap50']:.4f} smokeAP={d['smoke_ap50']:.4f}")
    # 알려진 값과 합쳐 2x2 매트릭스 구성(SSOT: 새 값만 계산, 나머지는 기존 기록 상수)
    matrix = {
        "note": "행=학습 도메인, 열=평가 도메인. mAP@0.5.",
        "AIHub_C4_11n": {"AIHub_val": 0.913, "DFire_test": 0.262},   # 기존 R8-보강 / A-1
        "Dfull": {"DFire_test": {"11m": 0.789, "11s": 0.787},        # 기존 D_full
                  "AIHub_val": {r["model"].replace("Dfull_", ""): round(r["map50"], 4) for r in rows}},
        "raw_dfull_on_aihub": rows,
    }
    (Path(PROJECT) / "domain_matrix.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[저장] runs/domain_matrix.json")


if __name__ == "__main__":
    main()
