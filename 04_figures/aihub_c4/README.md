# AIHub 071751 C4 재현 결과 (R8-보강, 2026-06-24)

> 대규모 일반화 근거. 수치 SSOT = `02_data_ssot/TRAINING_LOG.md` R8-보강 섹션.
> ⚠️ DFire(`dfire_4cell/`)와는 데이터셋·평가셋이 달라 **직접 수치 비교 불가** — 별개 벤치마크.

## 셀
| 셀 | 모델 | 구성 | train | val |
|----|------|------|------|-----|
| AIHub_C4 | YOLO11n | 균형(FL:SM 1:1)+NM | 114,462 | 19,080(자연분포) |

## 결과 (AIHub val)
mAP@0.5 = **0.913** (원래 E01 0.911 재현) / smoke AP 0.896 / fire AP 0.930 / P 0.905 / R 0.852.

## 파일
- `AIHub_C4_val_confusion_matrix.png` / `_norm.png` — val 혼동행렬
- `AIHub_C4_val_PR_curve.png` — val PR 곡선
- `AIHub_C4_train_results.png` — 학습 곡선
- `aihub_c4_summary.json` — 수치 원본

## 의미
"균형+NM" 구성이 114K 대규모에서도 0.913 → 구성 원리가 규모·데이터셋 무관하게 성립.
DFire = 통제 ablation(그림 완비), AIHub = 대규모 일반화 근거. 논문에서 별개 벤치마크로 병기.
