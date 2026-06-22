"""YOLO 파인튜닝 스크립트 — 단일 실험 또는 매트릭스 학습 지원.

단일 실험:
    python models/train.py \
        --model yolo11n \
        --data models/aihub_dataset/data.yaml \
        --epochs 100 --batch 64 --device 0

매트릭스 학습 (E01~E07):
    python models/train.py --matrix \
        --data models/aihub_dataset/data.yaml \
        --device 0

체크포인트 재개 (Spot VM 중단 후):
    python models/train.py --resume runs/yolo11n_E01/weights/last.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 매트릭스 실험 정의 (계획서 §C-1)
EXPERIMENT_MATRIX = [
    # (id, model, epochs, batch, imgsz, name_suffix)
    ("E01", "yolo11n.pt",  100, 64, 640, "E01_11n_base"),
    ("E02", "yolo11s.pt",  100, 48, 640, "E02_11s_base"),
    ("E03", "yolo11m.pt",  100, 32, 640, "E03_11m_server"),
    ("E04", "yolo11l.pt",   80, 16, 640, "E04_11l_server"),
    ("E05", "yolo11n.pt",  100, 64, 640, "E05_11n_synth"),   # 합성 데이터 사용 시 --data_synth
    ("E06", "yolo11s.pt",  100, 48, 640, "E06_11s_indoor"),  # 실내 데이터 사용 시 --data_indoor
    ("E07", "yolo11n.pt",  100, 64, 416, "E07_11n_416"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 fire/smoke 학습")
    parser.add_argument("--model",    type=str, default="yolo11n.pt",
                        help="모델 가중치 (yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11l.pt)")
    parser.add_argument("--data",     type=str, default="models/aihub_dataset/data.yaml",
                        help="데이터셋 YAML 경로")
    parser.add_argument("--data_synth",  type=str, default="",
                        help="합성 데이터 YAML (E05용)")
    parser.add_argument("--data_indoor", type=str, default="",
                        help="실내 특화 YAML (E06용)")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=64,
                        help="배치 크기 (-1 = 자동)")
    parser.add_argument("--device",   type=str, default="0")
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--project",  type=str, default="runs",
                        help="결과 저장 루트 디렉토리")
    parser.add_argument("--name",     type=str, default="",
                        help="실험 이름 (기본: 모델명 기반 자동 생성)")
    parser.add_argument("--save_period", type=int, default=5,
                        help="체크포인트 저장 간격 (에폭, Spot VM 대비)")
    parser.add_argument("--resume",   type=str, default="",
                        help="재개할 last.pt 경로 (Spot 중단 복구)")
    parser.add_argument("--matrix",   action="store_true",
                        help="매트릭스 전체 실험 실행 (E01~E07)")
    parser.add_argument("--matrix_ids", type=str, default="",
                        help="특정 실험만 실행 (예: E01,E03,E07)")
    parser.add_argument("--output_model", type=str, default="",
                        help="최종 best.pt 복사 경로")
    return parser.parse_args()


def run_single_experiment(
    model_path: str,
    data: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    project: str,
    name: str,
    save_period: int,
    resume_path: str = "",
    output_model: str = "",
) -> dict:
    """단일 실험 실행.

    Returns:
        결과 딕셔너리 {"name", "best_pt", "metrics_json", "elapsed_sec"}
    """
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DISABLED", "true")

    from ultralytics import YOLO

    t0 = time.time()

    if resume_path and Path(resume_path).exists():
        logger.info(f"[train] 재개: {resume_path}")
        model = YOLO(resume_path)
        results = model.train(resume=True)
    else:
        if not Path(model_path).exists():
            logger.info(f"[train] 모델 다운로드 중: {model_path}")
        model = YOLO(model_path)
        logger.info(
            f"[train] {name} 시작 — model={model_path}, data={data}, "
            f"epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}"
        )
        results = model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,           # -1: VRAM 60% 기준 자동 결정 (A100 80GB 권장)
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            save_period=save_period,
            patience=30,           # 조기 종료 (30에폭 개선 없으면 중단)
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,           # 코사인 LR 감쇠: 수렴 안정성 + 속도 향상
            warmup_epochs=3,
            cache=False,           # 디스크 캐시 비활성화 (NV 200GB 용량 초과 방지)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.0,
            verbose=True,
        )

    elapsed = time.time() - t0
    best_pt = Path(project) / name / "weights" / "best.pt"
    last_pt = Path(project) / name / "weights" / "last.pt"

    # 결과 메트릭 저장
    metrics_json = Path(project) / name / "metrics.json"
    metrics = {}
    if best_pt.exists():
        try:
            val_model = YOLO(str(best_pt))
            val_results = val_model.val(data=data, device=device, verbose=False)
            metrics = {
                "map50": float(val_results.box.map50),
                "map50_95": float(val_results.box.map),
                "precision": float(val_results.box.mp),
                "recall": float(val_results.box.mr),
                "per_class_ap50": [float(v) for v in val_results.box.ap50],
            }
            metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
            logger.info(
                f"[train] {name} 완료 — mAP50={metrics['map50']:.4f}, "
                f"elapsed={elapsed/3600:.1f}h"
            )
        except Exception as e:
            logger.warning(f"[train] 평가 실패: {e}")

    # output_model 복사
    if output_model and best_pt.exists():
        shutil.copy(best_pt, output_model)
        logger.info(f"[train] best.pt → {output_model}")

    return {
        "name": name,
        "best_pt": str(best_pt),
        "last_pt": str(last_pt),
        "metrics": metrics,
        "elapsed_sec": elapsed,
    }


def run_matrix(args: argparse.Namespace) -> None:
    """매트릭스 전체 실험 실행 (E01~E07)."""
    target_ids = set(args.matrix_ids.split(",")) if args.matrix_ids else None

    results_all: list[dict] = []
    summary_path = Path(args.project) / "matrix_summary.json"

    # 기존 완료 결과 로드 (재시작 시 건너뜀)
    completed_names: set[str] = set()
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            results_all = existing
            completed_names = {r["name"] for r in existing}
            logger.info(f"[matrix] 기존 결과 {len(completed_names)}개 로드 (건너뜀)")
        except Exception:
            pass

    for exp_id, model_pt, epochs, batch, imgsz, name in EXPERIMENT_MATRIX:
        if target_ids and exp_id not in target_ids:
            continue
        if name in completed_names:
            logger.info(f"[matrix] {name} 이미 완료 — 건너뜀")
            continue

        # E05: 합성 데이터
        data = args.data
        if exp_id == "E05" and args.data_synth:
            data = args.data_synth
            logger.info(f"[matrix] E05: 합성 데이터 사용 — {data}")
        # E06: 실내 데이터
        elif exp_id == "E06" and args.data_indoor:
            data = args.data_indoor
            logger.info(f"[matrix] E06: 실내 데이터 사용 — {data}")

        # 재개 체크
        last_pt = Path(args.project) / name / "weights" / "last.pt"
        resume = str(last_pt) if last_pt.exists() else ""
        if resume:
            logger.info(f"[matrix] {name}: 체크포인트에서 재개")

        result = run_single_experiment(
            model_path=model_pt,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=args.device,
            project=args.project,
            name=name,
            save_period=args.save_period,
            resume_path=resume,
        )
        results_all.append(result)
        completed_names.add(name)

        # 중간 저장
        summary_path.write_text(json.dumps(results_all, indent=2, ensure_ascii=False))

    # 최종 요약 출력
    logger.info("\n===== 매트릭스 학습 결과 요약 =====")
    best_map50 = 0.0
    best_result = None
    for r in results_all:
        map50 = r.get("metrics", {}).get("map50", 0.0)
        logger.info(
            f"  {r['name']}: mAP50={map50:.4f}, "
            f"elapsed={r.get('elapsed_sec', 0)/3600:.1f}h"
        )
        if map50 > best_map50:
            best_map50 = map50
            best_result = r

    if best_result:
        logger.info(f"\n[matrix] 최고 모델: {best_result['name']} (mAP50={best_map50:.4f})")
        best_src = Path(best_result["best_pt"])
        if best_src.exists():
            best_dst = Path(args.project) / "best_model" / "weights" / "best.pt"
            best_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_src, best_dst)
            logger.info(f"[matrix] 최고 모델 저장: {best_dst}")

    logger.info(f"[matrix] 요약 저장: {summary_path}")


def main() -> None:
    args = parse_args()

    if args.matrix:
        run_matrix(args)
        return

    # 단일 실험
    name = args.name
    if not name:
        model_stem = Path(args.model).stem
        name = f"{model_stem}_train"

    run_single_experiment(
        model_path=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=name,
        save_period=args.save_period,
        resume_path=args.resume,
        output_model=args.output_model,
    )


if __name__ == "__main__":
    main()
