"""데이터 구성 → 성능 정량 분해모델 (교수님 피드백 대응: 수식·크기분리·기여).

기존 통제 실험 데이터포인트만 사용(재학습 없음). 산출: runs/composition_model.json + 콘솔.
① 회귀식: mAP ~ balance(smoke박스비율) + NM + log(size) + capacity + dataset FE  → 표준화 계수
② 효과 분해표(데이터셋별, mean±std) + 지배비율(구성/모델)
③ 크기통제 NM 증명(E09/E10: 동일 총량 7,664)
클래스박스: fire 고정(D-Fire 9,638 / AIHub 57,391 fire-content) → balance는 크기 아닌 '구성' 변수.
"""
from __future__ import annotations
import json
import numpy as np
import statsmodels.api as sm

# (dataset, cell, smoke_box_frac, nm, size, capacity_M, mAP, std)  — SSOT 인용
ROWS = [
    # D-Fire (R8 test; C3/C4/C4_11s = multiseed mean±std, C1/C2 = single)
    ("DFire", "C1",     0.000, 0, 3828,  2.6, 0.325,  None),
    ("DFire", "C2",     0.032, 0, 4101,  2.6, 0.455,  None),
    ("DFire", "C3",     0.312, 0, 4598,  2.6, 0.6853, 0.0053),
    ("DFire", "C4",     0.312, 1, 11056, 2.6, 0.7395, 0.0044),
    ("DFire", "C4_11s", 0.312, 1, 11056, 9.4, 0.7508, 0.0021),
    # AIHub (shared val 19,080; multiseed mean±std)
    ("AIHub", "C1",     0.000, 0, 57391,  2.6, 0.4622, 0.0019),
    ("AIHub", "C2",     0.067, 0, 61490,  2.6, 0.7726, 0.0033),
    ("AIHub", "C3",     0.336, 0, 85767,  2.6, 0.9082, 0.0009),
    ("AIHub", "C4",     0.336, 1, 114462, 2.6, 0.9142, 0.0034),
]
# E09/E10 — 동일 총량 7,664 크기통제 (D-Fire test, multiseed mean)
VOLCTRL = {"C3_vol_signal": 0.720, "C4_eq_NM": 0.734, "size": 7664}  # E09/E10 SSOT 단일시드 일치(+1.4%p)


def zscore(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / (s if s > 0 else 1.0)


def main():
    ds = np.array([1.0 if r[0] == "AIHub" else 0.0 for r in ROWS])
    bal = np.array([r[2] for r in ROWS])
    nm = np.array([float(r[3]) for r in ROWS])
    logN = np.log(np.array([r[4] for r in ROWS], float))
    cap = np.array([r[5] for r in ROWS])
    y = np.array([r[6] for r in ROWS])

    # ── ① 표준화 회귀 (계수 크기 비교) ──
    X = np.column_stack([zscore(bal), zscore(nm), zscore(logN), zscore(cap), ds])
    Xc = sm.add_constant(X)
    names = ["const", "balance", "NM", "log_size", "capacity", "dataset(AIHub)"]
    m = sm.OLS(zscore(y), Xc).fit()
    coef = {n: round(float(c), 3) for n, c in zip(names, m.params)}
    pval = {n: round(float(p), 4) for n, p in zip(names, m.pvalues)}

    # 증분 R² — 올바른 순서: 데이터셋 baseline → 구성 → 크기(구성 후 잔여) → 모델
    #   size를 구성 뒤에 넣어, 구성이 이미 설명한 뒤 크기가 '추가로' 설명하는 몫을 봄(공선성 정직 처리).
    def r2(cols):
        return sm.OLS(y, sm.add_constant(np.column_stack(cols))).fit().rsquared
    r2_ds = r2([ds])
    r2_comp = r2([ds, bal, nm])
    r2_size = r2([ds, bal, nm, logN])
    r2_full = r2([ds, bal, nm, logN, cap])
    incr = {"dataset_only": round(r2_ds, 3),
            "+composition": round(r2_comp - r2_ds, 3),
            "+size_after_comp": round(r2_size - r2_comp, 3),
            "+model": round(r2_full - r2_size, 3),
            "total_R2": round(r2_full, 3)}
    # 공선성 진단: size vs dataset (같은 방향이면 분리 불가 → factorial 필요)
    corr_size_ds = round(float(np.corrcoef(logN, ds)[0, 1]), 3)

    # ── ② 효과 분해 (데이터셋별, %p) ──
    def cell(dset, name):
        for r in ROWS:
            if r[0] == dset and r[1] == name:
                return r[6]
        return None
    decomp = {}
    for dset in ("DFire", "AIHub"):
        c1, c3, c4 = cell(dset, "C1"), cell(dset, "C3"), cell(dset, "C4")
        d = {"balance_C1toC3": round((c3 - c1) * 100, 1), "NM_C3toC4": round((c4 - c3) * 100, 1)}
        if dset == "DFire":
            d["NM_size_controlled"] = round((VOLCTRL["C4_eq_NM"] - VOLCTRL["C3_vol_signal"]) * 100, 1)
            d["model_11n_to_11s"] = round((cell(dset, "C4_11s") - c4) * 100, 1)
        else:
            d["model_11n_to_11s_R7"] = 0.7  # R7 175K 별도 데이터(참고)
        me = d.get("model_11n_to_11s", d.get("model_11n_to_11s_R7"))
        d["dominance_comp_over_model"] = round(d["balance_C1toC3"] / me, 1) if me else None
        decomp[dset] = d

    out = {
        "n_points": len(ROWS),
        "standardized_regression": {"coef": coef, "pvalue": pval, "R2": round(m.rsquared, 3),
                                    "note": "⚠ 풀드 β는 size↔dataset 공선성으로 개별해석 불가(size가 baseline 흡수). 증분R²·분해표를 볼 것."},
        "collinearity_size_dataset": {"corr": corr_size_ds,
                                      "note": f"log(size)와 dataset 상관 {corr_size_ds} → 거의 완전공선. 풀드 회귀로 크기효과 분리 불가 → factorial 필요."},
        "incremental_R2": incr,
        "decomposition_pp": decomp,
        "size_controlled_NM_E09E10": {"equal_size": VOLCTRL["size"],
                                      "signal_add": VOLCTRL["C3_vol_signal"], "NM_add": VOLCTRL["C4_eq_NM"],
                                      "NM_gain_pp": round((VOLCTRL["C4_eq_NM"] - VOLCTRL["C3_vol_signal"]) * 100, 1)},
    }
    (__import__("pathlib").Path("runs/composition_model.json")).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 콘솔 요약 ──
    print("=== ① 표준화 회귀 (mAP ~ balance+NM+log_size+capacity+dataset) ===")
    for n in names[1:]:
        print(f"  β[{n:14s}] = {coef[n]:+.3f}   (p={pval[n]})")
    print(f"  R² = {out['standardized_regression']['R2']}  ⚠ size↔dataset 공선(corr={corr_size_ds}) → 개별 β 해석불가")
    print("=== 증분 R² (데이터셋→구성→크기→모델 순) ===")
    for k, v in incr.items():
        print(f"  {k}: {v}")
    print(f"  → 구성이 데이터셋 위에 +{incr['+composition']}, 그 뒤 크기는 +{incr['+size_after_comp']}(거의 0), 모델 +{incr['+model']}")
    print("=== ② 효과 분해 (%p) ===")
    for dset, d in decomp.items():
        print(f"  [{dset}] {d}")
    v = out["size_controlled_NM_E09E10"]
    print(f"=== ③ 크기통제 NM (동일 {v['equal_size']}): 신호 {v['signal_add']} vs NM {v['NM_add']} → NM +{v['NM_gain_pp']}%p ===")
    print("[저장] runs/composition_model.json")


if __name__ == "__main__":
    main()
