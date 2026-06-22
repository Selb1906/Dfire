# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is **not a running software project** — it is a paper-writing workspace. A failed national R&D proposal (EXPERT, Jetson-based autonomous fire-safety agent) is being converted into a KCI journal paper out of an already-completed fire/smoke detection ablation study. The repository holds the manuscript draft, the numeric source-of-truth, and reproduction scripts.

**Central thesis of the paper**: fire-detection performance is determined by *data composition* (class balance + normal-background inclusion), not model size. Class balance (FL:SM 1:1) +7.5%p; normal background (NM) +7.5%p; model capacity (11n→11s) +0.7%p — so data factors outweigh model factors ~10×. Final: YOLO11n, mAP@0.5 = 0.911, Jetson Orin Nano TensorRT FP16 ~47 FPS.

Start by reading `00_인계문서_현황과할일.md` (status + to-do handoff) — it is the entry point that explains the whole situation.

## Layout

| Path | Role |
|------|------|
| `00_인계문서_현황과할일.md` | Handoff doc (current status + to-do) — read first |
| `01_manuscript/KCI_화재감지_데이터구성_ablation_v1.md` | Full manuscript draft (abstract→conclusion) |
| `01_manuscript/투고준비_체크리스트.md` | Submission checklist: what's ready now vs. needs figures; candidate journals |
| `02_data_ssot/TRAINING_LOG.md` | **SSOT for every number in the paper** |
| `03_scripts/` | Reproduction scripts (training/conversion/edge export) |
| `04_figures/` | Reference images only — **not** the paper's result figures (read `04_figures/README.md`) |

## Hard rules specific to this repo

- **`02_data_ssot/TRAINING_LOG.md` is the single source of truth for all metrics.** Every table number in the manuscript must trace back to it. When experiments are re-run, update TRAINING_LOG *first*, then the manuscript tables — never invent or adjust a number directly in the manuscript.
- **Trained weights and run artifacts were deleted** (training data 137–175K images, `best.pt` E01–E07, confusion matrices / PR curves / `results.csv`). Consequence: tables and text are completable now, but result *figures* (confusion matrix, PR curve, qualitative detections) cannot be regenerated without retraining. Bar charts of mAP-by-composition *can* be drawn directly from TRAINING_LOG numbers (no weights needed).
- **Test-set discontinuity**: R1 used the DFire test set; R2 onward used the AIHub test set. These numbers are **not directly comparable** — never present R1 (0.786) in the same comparison series as R2+ without noting the different evaluation set.
- **Data is public and clean.** The study uses AIHub 071751 (public, no NDA, no privacy concern). No field/실증 data (e.g. care-home) is included, so the `private-data-disclosure` rule has zero violations here — keep it that way; do not introduce any private/field dataset references into the manuscript.
- `04_figures/arch_diagram.png` is from the *rejected EXPERT proposal* and is unrelated to this paper — do not put it in the manuscript.

## Reproduction pipeline (only when retraining — see handoff §6, option B)

Run in order; all scripts use `ultralytics` (YOLO11) and assume a CUDA GPU (training was RTX 4090 local / RunPod L40S cloud):

1. `aihub_to_yolo.py` — AIHub raw → YOLO format; includes video scene sampling (histogram scene-cut + 2s interval + phash dedup) and FL:SM:NM balancing (`--balance_nm`).
2. `train.py` — single experiment, or `--matrix` / `--matrix_ids E01,E03` for the E01–E07 grid. `train_matrix.py` is the cloud multi-Pod orchestrator (checkpoint resume for Spot interruption).
3. `evaluate.py` — `model.val()` metrics (warns if smoke recall < 0.8).
4. `export_trt.py` — TensorRT FP16 export + FPS bench. **Jetson Orin Nano only** — does not run on Windows.

The "core 4-cell retrain" recommended in the handoff uses Kaggle DFire (~21K imgs, ~6–12h on a 4090) to regenerate all figures + FP/FPR as a single clean public benchmark. Note: retraining changes absolute numbers, so the manuscript tables must be updated to match.

## Conventions

- All documents are Korean (UTF-8). Preserve terminology and names; edit only what is requested.
- Not a git repository.
