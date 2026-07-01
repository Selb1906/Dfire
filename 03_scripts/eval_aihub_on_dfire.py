"""요청 A-1 — AIHub C4 체크포인트를 DFire test(4,306)로 크로스도메인 평가 (재학습 없음).

목적: Table 6 선행연구(DFire test 기준)와 직접 비교 가능한 수치 확보.
AIHub C4(runs/AIHub_C4, YOLO11n, 균형+NM, AIHub val 0.913) 가중치로 DFire test 평가.
클래스 0=smoke, 1=fire 양쪽 통일이라 그대로 평가 가능.
"""
from __future__ import annotations
import json, os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
BEST = BASE / "runs" / "AIHub_C4" / "weights" / "best.pt"
DFIRE_YAML = str(BASE / "compositions" / "c4_balanced_nm.yaml")  # test → dfire/data/test


def metrics_dict(res) -> dict:
    ap50 = [float(v) for v in res.box.ap50]; r = [float(v) for v in res.box.r]
    return {"map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "smoke_ap50": ap50[0] if ap50 else None,
            "fire_ap50": ap50[1] if len(ap50) > 1 else None,
            "smoke_recall": r[0] if r else None,
            "fire_recall": r[1] if len(r) > 1 else None}


def main():
    from ultralytics import YOLO
    assert BEST.exists(), f"AIHub C4 best.pt 없음: {BEST}"
    print(f"[eval] AIHub C4 → DFire test 평가: {BEST}")
    model = YOLO(str(BEST))
    res = model.val(data=DFIRE_YAML, split="test", imgsz=640, device="0",
                    project=PROJECT, name="AIHub_C4_on_DFiretest", exist_ok=True,
                    plots=True, verbose=False)
    out = {"model": "AIHub_C4 (YOLO11n, AIHub 114K 균형+NM)",
           "eval_set": "DFire test (4,306)", "test": metrics_dict(res)}
    (Path(PROJECT) / "aihub_c4_on_dfiretest.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    t = out["test"]
    print(f"\n[결과] AIHub C4 on DFire test:")
    print(f"  mAP@0.5={t['map50']:.4f}  mAP@0.5:0.95={t['map50_95']:.4f}")
    print(f"  P={t['precision']:.4f}  R={t['recall']:.4f}")
    print(f"  smoke AP={t['smoke_ap50']:.4f}  fire AP={t['fire_ap50']:.4f}")
    print(f"  (참고) DFire 단독 C4 test mAP@0.5 = 0.740 (멀티시드 평균)")


if __name__ == "__main__":
    main()
