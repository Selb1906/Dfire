#!/bin/bash
# Jetson에서 실행 — 분석세션 머신에서 best.pt 가져오기
# ANALYSIS_IP와 ANALYSIS_USER를 실제 값으로 수정 후 실행

ANALYSIS_IP="192.168.x.x"       # 분석세션 머신 IP로 변경
ANALYSIS_USER="username"          # 분석세션 머신 사용자명으로 변경

mkdir -p weights

# R8 C4 (YOLO11n @640) — 정확도 우선 (test mAP 0.736)
scp "${ANALYSIS_USER}@${ANALYSIS_IP}:C:/YangHyunHo/DFire/runs/E_C4/weights/best.pt" \
    weights/best_C4_640.pt

# E08 C4 (YOLO11n @416) — Jetson FPS 최적 (test mAP 0.700, 처리량 ~2배)
scp "${ANALYSIS_USER}@${ANALYSIS_IP}:C:/YangHyunHo/DFire/runs/E08_C4_416/weights/best.pt" \
    weights/best_C4_416.pt

echo "완료. weights/ 폴더 확인:"
ls -lh weights/

# 변환 시 주의: export imgsz를 학습값과 일치 (640 / 416). 클래스 0=smoke, 1=fire.
