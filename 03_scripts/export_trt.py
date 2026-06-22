"""TensorRT FP16 변환 스크립트 (Jetson Orin Nano 전용).

단일 모델 변환:
    python models/export_trt.py \
        --model models/best.pt \
        --output models/best.engine

배치 변환 (매트릭스 학습 결과 전체):
    python models/export_trt.py \
        --batch \
        --runs_dir runs \
        --output_dir models/engines

이 스크립트는 Jetson에서 실행되어야 합니다.
Windows에서는 TensorRT 및 ultralytics CUDA 익스포트가 동작하지 않습니다.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error(
        "ultralytics 패키지를 찾을 수 없습니다.\n"
        "Jetson JetPack 환경에서 다음 명령으로 설치하세요:\n"
        "  pip install ultralytics\n"
        "Windows에서는 TensorRT FP16 변환이 지원되지 않습니다.\n"
        "반드시 Jetson Orin Nano에서 실행하세요."
    )
    sys.exit(1)

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO 모델을 TensorRT FP16 엔진으로 변환합니다 (Jetson 전용)."
    )
    # 단일 모드
    parser.add_argument("--model",  default="models/best.pt",
                        help="입력 모델 경로 (단일 변환 시)")
    parser.add_argument("--output", default="",
                        help="출력 엔진 경로 (단일 변환 시, 기본: 모델명.engine)")
    # 배치 모드
    parser.add_argument("--batch",  action="store_true",
                        help="배치 모드: runs_dir 내 모든 best.pt 변환")
    parser.add_argument("--runs_dir",    default="runs",
                        help="배치 모드: 학습 결과 루트 (runs/)")
    parser.add_argument("--output_dir",  default="models/engines",
                        help="배치 모드: 변환된 .engine 저장 폴더")
    # 공통
    parser.add_argument("--device", default="0",
                        help="GPU device ID (기본: 0)")
    parser.add_argument("--imgsz",  type=int, default=640,
                        help="추론 이미지 크기 (기본: 640)")
    parser.add_argument("--force",  action="store_true",
                        help="출력 파일이 이미 존재해도 덮어씁니다.")
    parser.add_argument("--benchmark", action="store_true",
                        help="변환 후 FPS 벤치마크 실행 (100프레임)")
    return parser.parse_args()


def _find_exported_engine(model_path: str) -> Path | None:
    """ultralytics export 후 자동 생성된 .engine 파일을 찾습니다."""
    model_p = Path(model_path)
    candidate = model_p.with_suffix(".engine")
    if candidate.exists():
        return candidate
    for pattern in ["runs/export/*/*.engine", "runs/export/*/*/*.engine"]:
        matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def convert_single(
    model_path: str,
    output_path: str,
    device: str,
    imgsz: int,
    force: bool,
    run_benchmark: bool,
) -> dict:
    """단일 .pt → .engine 변환.

    Returns:
        결과 딕셔너리 {"model", "engine", "fps", "success"}
    """
    out = Path(output_path) if output_path else Path(model_path).with_suffix(".engine")

    if out.exists() and not force:
        logger.warning(f"[export] 이미 존재: {out}. 건너뜀 (--force로 덮어씁니다)")
        return {"model": model_path, "engine": str(out), "fps": None, "success": True, "skipped": True}

    out.parent.mkdir(parents=True, exist_ok=True)
    model_p = Path(model_path)
    if not model_p.exists():
        logger.error(f"[export] 모델 파일 없음: {model_p}")
        return {"model": model_path, "engine": str(out), "fps": None, "success": False}

    logger.info(f"[export] 변환 시작: {model_p} → {out} (imgsz={imgsz}, FP16)")
    model = YOLO(str(model_p))
    model.export(format="engine", half=True, device=device, imgsz=imgsz)

    engine_src = _find_exported_engine(str(model_p))
    if engine_src is None:
        logger.error(f"[export] .engine 파일을 찾을 수 없음")
        return {"model": model_path, "engine": str(out), "fps": None, "success": False}

    if engine_src.resolve() != out.resolve():
        shutil.copy2(str(engine_src), str(out))

    logger.info(f"[export] 변환 완료: {out}")

    # 추론 테스트
    trt_model = YOLO(str(out))
    dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    trt_model.predict(dummy_frame, verbose=False)
    logger.info(f"[export] 추론 테스트 OK")

    fps = None
    if run_benchmark:
        logger.info("[export] FPS 벤치마크 (100프레임)...")
        t0 = time.perf_counter()
        for _ in range(100):
            trt_model.predict(dummy_frame, verbose=False)
        elapsed = time.perf_counter() - t0
        fps = round(100 / elapsed, 1)
        logger.info(f"[export] FPS: {fps:.1f}")

    return {"model": model_path, "engine": str(out), "fps": fps, "success": True}


def run_batch(
    runs_dir: str,
    output_dir: str,
    device: str,
    imgsz: int,
    force: bool,
    run_benchmark: bool,
) -> None:
    """runs/ 내 모든 best.pt를 배치 변환.

    출력: {output_dir}/{exp_name}.engine
    요약: {output_dir}/export_summary.json
    """
    runs_p = Path(runs_dir)
    out_p  = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    best_pts = sorted(runs_p.glob("*/weights/best.pt"))
    logger.info(f"[batch] {len(best_pts)}개 모델 발견")

    results = []
    for pt in best_pts:
        exp_name = pt.parent.parent.name  # runs/{exp_name}/weights/best.pt
        out_engine = out_p / f"{exp_name}.engine"

        result = convert_single(
            model_path=str(pt),
            output_path=str(out_engine),
            device=device,
            imgsz=imgsz,
            force=force,
            run_benchmark=run_benchmark,
        )
        result["exp_name"] = exp_name
        results.append(result)

    # 요약 저장
    summary = out_p / "export_summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    success = sum(1 for r in results if r.get("success"))
    logger.info(f"\n[batch] 완료: {success}/{len(results)} 성공")
    if run_benchmark:
        logger.info("[batch] FPS 요약:")
        for r in results:
            if r.get("fps"):
                logger.info(f"  {r['exp_name']}: {r['fps']:.1f} FPS")
    logger.info(f"[batch] 요약 저장: {summary}")


def main() -> None:
    args = parse_args()

    if args.batch:
        run_batch(
            runs_dir=args.runs_dir,
            output_dir=args.output_dir,
            device=args.device,
            imgsz=args.imgsz,
            force=args.force,
            run_benchmark=args.benchmark,
        )
    else:
        convert_single(
            model_path=args.model,
            output_path=args.output,
            device=args.device,
            imgsz=args.imgsz,
            force=args.force,
            run_benchmark=args.benchmark,
        )


if __name__ == "__main__":
    main()
