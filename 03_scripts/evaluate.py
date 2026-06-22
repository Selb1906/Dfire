"""모델 성능 평가 스크립트."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11n smoke/fire 모델 평가")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n_smoke.pt",
        help="평가할 모델 가중치 경로",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="models/smoke_dataset/data.yaml",
        help="데이터셋 YAML 경로",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="입력 이미지 크기",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="평가 디바이스 (0, cpu 등)",
    )
    return parser.parse_args()


def evaluate(model_path: str, data: str, imgsz: int, device: str) -> dict:
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[evaluate] 오류: 모델 파일을 찾을 수 없습니다 — {model_path}")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"[evaluate] 모델 로드: {model_path}")
    model = YOLO(model_path)

    print(f"[evaluate] 평가 시작 — data={data}, imgsz={imgsz}, device={device}")
    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=True)

    map50 = metrics.box.map50
    recall_smoke = metrics.box.r[0] if len(metrics.box.r) > 0 else 0.0

    if recall_smoke < 0.8:
        print("WARNING: smoke recall < 0.8. 임계값 0.4로 조정 권장.")
        recommended_threshold = 0.4
    else:
        print("OK: 연기 recall >= 0.8 달성.")
        recommended_threshold = 0.6

    result = {
        "map50": map50,
        "smoke_recall": recall_smoke,
        "recommended_threshold": recommended_threshold,
    }
    print(f"[evaluate] 결과: {result}")
    return result


def main() -> None:
    args = parse_args()
    evaluate(
        model_path=args.model,
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
