# DFire 4셀 재학습 결과 그림 (R8, 2026-06-22)

> 출처: RTX 5090 로컬 재학습. 수치 SSOT = `02_data_ssot/TRAINING_LOG.md` R8 섹션.
> 가중치(best.pt)·전체 산출물은 `runs/`(git 비추적, 로컬 보존). 여기엔 논문용 그림만 추렸다.

## 셀 정의 (5셀)
| 셀 | 모델 | 구성 |
|----|------|------|
| C1 | YOLO11n | fire-only (smoke 미학습) |
| C2 | YOLO11n | 불균형 14:1 (소량 smoke) |
| C3 | YOLO11n | 균형 1:1 |
| C4 | YOLO11n | 균형 + 정상배경(NM) |
| C4_11s | YOLO11s | C4 동일 데이터, 모델만 변경 |

## 논문용 그림 (publication — 영문 전용, 제목 없음)
- `fig_data_composition_dfire.png` — 데이터 구성별 test mAP@0.5 + smoke AP. **C3/C4는 멀티시드 n=3 평균±std 오차막대**(C3 0.685±0.005, C4 0.740±0.004), C1/C2는 단일시드. 화살표 +36.0%p/+5.5%p. 300dpi
- `fig_model_dfire.png` — 모델 용량 mAP@0.5 (**n=3 평균±std**: 11n 0.740±0.004, 11s 0.751±0.002, +1.1%p), 오차막대. 300dpi
- `fig_smoke_ap_trend_dfire.png` — 구성별 test smoke AP 막대 (단일시드 0.000/0.237/0.698/0.765 — 멀티시드 smoke AP 미집계), 300dpi
- `C{1..4}_test_confusion_matrix_norm.png` — 셀별 정규화 혼동행렬 (test)
- `fig_confusion_matrix_comp_dfire.png` — C1~C4 정규화 혼동행렬 **2×2 그리드** (단일 컬러바, 데이터 재계산, `make_confusion_grid.py`)
- `C4_test_PR_curve.png` — C4 PR 곡선 (fire/smoke 클래스별)
- `fig_qualitative_success.png` — C4 정성 성공례 (화염 0.83 + 연기 0.84), `make_qualitative.py`
- `fig_qualitative_failure.png` — C4 정성 한계례 (정상배경 오탐 + 원거리 화염 미탐)
- 생성: `03_scripts/plot_dfire_ablation.py` (수치 기반, GPU 불요)

## 파일
- `{셀}_test_confusion_matrix.png` / `_norm.png` — **test 셋 혼동행렬**(논문 결과 그림)
- `{셀}_test_PR_curve.png` — test 셋 Precision–Recall 곡선
- `{셀}_train_results.png` — 학습 곡선(loss·metric 추이)
- `4cell_summary.json` — val·test 전체 수치 원본

## 핵심 (test mAP@0.5)
C1 0.325 → C2 0.455 → C3 0.691 → C4 0.736 → C4_11s 0.749.
균형 총효과(C1→C3) +36.6%p, NM(C3→C4) +4.5%p ≫ 모델(11n→11s) +1.3%p — 데이터 구성이 모델보다 지배적.
C1은 smoke AP=0(연기 미학습)으로 평균이 낮음(fire AP 단독 0.650).
⚠️ AIHub의 "C2<C1(소량 smoke=노이즈)"은 DFire에서 **재현 안 됨**(C2 0.455>C1, fire AP도 단조 증가). 상세 = TRAINING_LOG R8 주석.
