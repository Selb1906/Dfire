# DFire 4셀 재학습 결과 그림 (R8, 2026-06-22)

> 출처: RTX 5090 로컬 재학습. 수치 SSOT = `02_data_ssot/TRAINING_LOG.md` R8 섹션.
> 가중치(best.pt)·전체 산출물은 `runs/`(git 비추적, 로컬 보존). 여기엔 논문용 그림만 추렸다.

## 셀 정의
| 셀 | 모델 | 구성 |
|----|------|------|
| C1 | YOLO11n | fire-only (smoke 미학습) |
| C3 | YOLO11n | 균형 1:1 |
| C4 | YOLO11n | 균형 + 정상배경(NM) |
| C4_11s | YOLO11s | C4 동일 데이터, 모델만 변경 |

## 파일
- `{셀}_test_confusion_matrix.png` / `_norm.png` — **test 셋 혼동행렬**(논문 결과 그림)
- `{셀}_test_PR_curve.png` — test 셋 Precision–Recall 곡선
- `{셀}_train_results.png` — 학습 곡선(loss·metric 추이)
- `4cell_summary.json` — val·test 전체 수치 원본

## 핵심 (test mAP@0.5)
C1 0.325 → C3 0.691 (+36.6%p, 균형) → C4 0.736 (+4.5%p, NM) → C4_11s 0.749 (+1.3%p, 모델).
데이터 구성이 모델 크기보다 지배적. C1은 smoke AP=0(연기 미학습)으로 평균이 낮음 — fire AP 단독은 0.650.
