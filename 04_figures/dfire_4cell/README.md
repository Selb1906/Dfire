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

## 논문용 그림 (publication)
- `fig_data_composition_dfire.png` — 데이터 구성 단계별 test mAP@0.5 + smoke AP 추이 (11n 고정, C1~C4), 300dpi
- `fig_model_dfire.png` — 모델 용량 mAP@0.5 (C4: 11n vs 11s), 300dpi
- `C{1..4}_test_confusion_matrix_norm.png` — 셀별 정규화 혼동행렬 (test)
- `C4_test_PR_curve.png` — C4 PR 곡선 (fire/smoke 클래스별)
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
