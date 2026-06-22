#!/usr/bin/env python3
"""
RunPod 클라우드 매트릭스 학습 실행 스크립트

역할: 7개 실험(E01~E07)을 순차 실행, 체크포인트 자동 저장/재개.
Spot VM 중단 시 마지막 체크포인트에서 자동 재개.

── Pod 분할 전략 ──────────────────────────────────────────
  2-Pod (A100×2, 46h, $76):
    Pod 1: E01 E03 E05 E07  →  32.5h
    Pod 2: E02 E04 E06      →  46h ← 병목

  3-Pod (A100×3, 27h, $75):  ← 추천 (동일 비용, 41% 빠름)
    Pod 1: E04              →  27h ← 병목 격리
    Pod 2: E03 E06          →  26h
    Pod 3: E01 E02 E05 E07  →  25.5h

  4-Pod (A100×4, 16h, $76):  ← 최고속 (E03/E04 각각 분리)
    Pod 1: E04              →  27h → H100으로 ~16h
    Pod 2: E03              →  16h
    Pod 3: E02 E06          →  19h ← 병목
    Pod 4: E01 E05 E07      →  16.5h
──────────────────────────────────────────────────────────

사용법:
  python scripts/cloud/train_matrix.py --pod 1   # 3-Pod Pod1: E04
  python scripts/cloud/train_matrix.py --pod 2   # 3-Pod Pod2: E03 E06
  python scripts/cloud/train_matrix.py --pod 3   # 3-Pod Pod3: E01 E02 E05 E07
  python scripts/cloud/train_matrix.py --experiments E01,E03  # 직접 지정
  python scripts/cloud/train_matrix.py           # 전체 순차
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 클라우드 기본 경로
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
DATA_YAML       = str(WORKSPACE / "data" / "integrated" / "data.yaml")
DATA_SYNTH_YAML = str(WORKSPACE / "data" / "synth" / "data.yaml")
DATA_INDOOR_YAML = str(WORKSPACE / "data" / "indoor" / "data.yaml")
RUNS_DIR        = str(WORKSPACE / "runs")

# ── batch=-1 (auto) 적용 시 A100 80GB 추정 시간 ──────────────
# batch=-1: ultralytics가 VRAM 60% 기준 자동 결정
#   11n → batch≈192, 11s → batch≈128, 11m → batch≈96, 11l → batch≈64
# 배치 증가 배수: 11n×3, 11s×2.7, 11m×3, 11l×4 → 에폭당 처리 속도 동비율 향상
#
# 실험별 예상 시간 (batch=-1, A100 80GB):
#   E01(11n)  3h | E02(11s) 4.5h | E03(11m)  6h | E04(11l) 7h
#   E05(11n)  4h | E06(11s)   5h | E07(11n@416) 2.5h
#   총: ~32h / 3-Pod Wall-clock: ~11h

# ── 2-Pod 분할 (batch=-1, A100×2, Wall-clock ~16.5h, $53)
POD2_SPLIT = {
    1: {"E04", "E03", "E07"},          # 7+6+2.5 = 15.5h
    2: {"E02", "E01", "E05", "E06"},  # 4.5+3+4+5 = 16.5h ← 병목
}

# ── 3-Pod 분할 (batch=-1, A100×3, Wall-clock ~11h, $54) ← 추천
POD3_SPLIT = {
    1: {"E04", "E05"},                 # 7+4 = 11h
    2: {"E03", "E06"},                 # 6+5 = 11h
    3: {"E01", "E02", "E07"},         # 3+4.5+2.5 = 10h
}

# ── 4-Pod 분할 (batch=-1, A100×4, Wall-clock ~7.5h, $55)
POD4_SPLIT = {
    1: {"E04"},                        # 7h ← 단독
    2: {"E03", "E07"},                 # 6+2.5 = 8.5h ← 새 병목
    3: {"E02", "E05"},                 # 4.5+4 = 8.5h
    4: {"E01", "E06"},                 # 3+5 = 8h
}

# 실험 정의 (계획서 §C-1 완전 동기화)
EXPERIMENTS = {
    # batch=-1: A100 80GB VRAM 60% 자동 사용 (11n≈192, 11s≈128, 11m≈96, 11l≈64)
    "E01": dict(model="yolo11n.pt", epochs=100, batch=-1, imgsz=640, name="E01_11n_base",
                data=DATA_YAML, notes="Jetson 베이스라인"),
    "E02": dict(model="yolo11s.pt", epochs=100, batch=-1, imgsz=640, name="E02_11s_base",
                data=DATA_YAML, notes="Jetson 후보 (균형)"),
    "E03": dict(model="yolo11m.pt", epochs=100, batch=-1, imgsz=640, name="E03_11m_server",
                data=DATA_YAML, notes="서버 후보 1"),
    "E04": dict(model="yolo11l.pt", epochs=80,  batch=-1, imgsz=640, name="E04_11l_server",
                data=DATA_YAML, lr0=0.0005, lrf=0.005,
                notes="서버 후보 2 (최대 정확도, lr0=0.0005 안정화)"),
    "E05": dict(model="yolo11n.pt", epochs=100, batch=-1, imgsz=640, name="E05_11n_synth",
                data=DATA_SYNTH_YAML if Path(DATA_SYNTH_YAML).exists() else DATA_YAML,
                notes="합성 증강 효과"),
    "E06": dict(model="yolo11s.pt", epochs=100, batch=-1, imgsz=640, name="E06_11s_indoor",
                data=DATA_INDOOR_YAML if Path(DATA_INDOOR_YAML).exists() else DATA_YAML,
                notes="실내 파인튜닝 후보"),
    "E07": dict(model="yolo11n.pt", epochs=100, batch=-1, imgsz=416, name="E07_11n_416",
                data=DATA_YAML, notes="소형 입력 Jetson 최적화"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RunPod 매트릭스 학습")
    parser.add_argument("--pod",         type=int, choices=[0, 1, 2, 3, 4], default=0,
                        help="Pod 번호 (0=전체순차 / 1~3=3-Pod분할 / --pods 옵션으로 총 수 지정)")
    parser.add_argument("--pods",        type=int, choices=[2, 3, 4], default=3,
                        help="총 Pod 수 (2/3/4, 기본: 3 ← A100×3 추천)")
    parser.add_argument("--experiments", type=str, default="",
                        help="특정 실험 ID (쉼표 구분, 예: E01,E03)")
    parser.add_argument("--device",      type=str, default="0")
    parser.add_argument("--save_period", type=int, default=5,
                        help="체크포인트 저장 간격 (에폭)")
    parser.add_argument("--runs_dir",    type=str, default=RUNS_DIR)
    parser.add_argument("--dry_run",     action="store_true",
                        help="실제 학습 없이 실험 목록만 출력")
    return parser.parse_args()


def select_experiments(args: argparse.Namespace) -> list[str]:
    """실행할 실험 ID 목록 반환.

    --pod 0          : 전체 순차
    --pod N --pods 2 : 2-Pod 분할
    --pod N --pods 3 : 3-Pod 분할 (기본, 추천)
    --pod N --pods 4 : 4-Pod 분할
    --experiments    : 직접 지정 (우선)
    """
    if args.experiments:
        return [e.strip().upper() for e in args.experiments.split(",")]
    if args.pod == 0:
        return list(EXPERIMENTS.keys())

    splits = {2: POD2_SPLIT, 3: POD3_SPLIT, 4: POD4_SPLIT}
    split = splits.get(args.pods, POD3_SPLIT)
    experiments = split.get(args.pod)
    if experiments is None:
        logger.error(f"[matrix] --pod {args.pod}은 --pods {args.pods} 범위 초과")
        return []
    return sorted(experiments)


def run_all(args: argparse.Namespace) -> None:
    from models.train import run_single_experiment

    exp_ids = select_experiments(args)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = runs_dir / "matrix_summary.json"
    results: list[dict] = []
    completed: set[str] = set()

    # 기존 요약 로드
    if summary_path.exists():
        try:
            results = json.loads(summary_path.read_text(encoding="utf-8"))
            completed = {r["name"] for r in results if r.get("success")}
            logger.info(f"[matrix] 기존 완료 실험: {completed}")
        except Exception:
            pass

    logger.info(f"[matrix] 실행 대상: {exp_ids}")
    if args.dry_run:
        for eid in exp_ids:
            exp = EXPERIMENTS[eid]
            logger.info(f"  {eid}: {exp['model']} epochs={exp['epochs']} batch={exp['batch']} imgsz={exp['imgsz']} — {exp['notes']}")
        return

    for eid in exp_ids:
        exp = EXPERIMENTS.get(eid)
        if exp is None:
            logger.warning(f"[matrix] 알 수 없는 실험 ID: {eid}")
            continue

        exp_name = exp["name"]
        if exp_name in completed:
            logger.info(f"[matrix] {eid} ({exp_name}) 이미 완료 — 건너뜀")
            continue

        # 체크포인트 재개 확인
        last_pt = Path(args.runs_dir) / exp_name / "weights" / "last.pt"
        resume = str(last_pt) if last_pt.exists() else ""
        if resume:
            logger.info(f"[matrix] {eid}: 체크포인트에서 재개 — {last_pt}")

        # 데이터 YAML 파일 존재 확인
        data_yaml = exp["data"]
        if not Path(data_yaml).exists():
            logger.warning(f"[matrix] {eid}: data YAML 없음 ({data_yaml}), 기본 사용: {DATA_YAML}")
            data_yaml = DATA_YAML

        t0 = time.time()
        try:
            result = run_single_experiment(
                model_path=exp["model"],
                data=data_yaml,
                epochs=exp["epochs"],
                batch=exp["batch"],
                imgsz=exp["imgsz"],
                device=args.device,
                project=args.runs_dir,
                name=exp_name,
                save_period=args.save_period,
                resume_path=resume,
            )
            result["exp_id"] = eid
            result["notes"] = exp["notes"]
            result["success"] = True
        except Exception as e:
            logger.error(f"[matrix] {eid} 실패: {e}")
            result = {
                "exp_id": eid, "name": exp_name, "notes": exp["notes"],
                "success": False, "error": str(e),
                "elapsed_sec": time.time() - t0,
            }

        results.append(result)
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # 결과 요약
    logger.info("\n" + "=" * 50)
    logger.info("매트릭스 학습 결과")
    logger.info("=" * 50)
    best = {"map50": 0.0, "name": ""}
    for r in results:
        map50 = r.get("metrics", {}).get("map50", 0.0)
        status = "✓" if r.get("success") else "✗"
        logger.info(
            f"  [{status}] {r['exp_id']} ({r['name']}): "
            f"mAP50={map50:.4f}, "
            f"elapsed={r.get('elapsed_sec', 0)/3600:.1f}h"
            f" — {r['notes']}"
        )
        if map50 > best["map50"]:
            best["map50"] = map50
            best["name"] = r["name"]

    if best["name"]:
        logger.info(f"\n최고 모델: {best['name']} (mAP50={best['map50']:.4f})")

    # 최고 모델 별도 저장
    if best["name"]:
        best_src = Path(args.runs_dir) / best["name"] / "weights" / "best.pt"
        if best_src.exists():
            best_dst = Path(args.runs_dir) / "best_model" / "weights" / "best.pt"
            best_dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(best_src, best_dst)
            logger.info(f"최고 모델 저장: {best_dst}")

    logger.info(f"요약: {summary_path}")


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
