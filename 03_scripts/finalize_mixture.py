"""혼합물설계 Phase A 마무리 — mixture_pilot.json → TRAINING_LOG 새 섹션 + push + 작업 자삭제.

셀별 mean±std(mAP@0.5/fire/smoke/P/R) 표. V1~V3은 절대 포함 안 함(총괄세션 별도 지시).
"""
from __future__ import annotations
import json, subprocess, sys, statistics as st
from pathlib import Path

BASE = Path(r"C:\YangHyunHo\DFire")
LOG = BASE / "02_data_ssot" / "TRAINING_LOG.md"
J = BASE / "runs" / "mixture_pilot.json"
ANCHOR = "## 모델 선정 기준"
COMP = {"P1": (1, 0, 0), "P2": (0, 1, 0), "P3": (0, 0, 1), "P4": (0.5, 0.5, 0),
        "P5": (0.5, 0, 0.5), "P6": (0, 0.5, 0.5), "P7": (1/3, 1/3, 1/3)}
NIMG = {"P1": "F6000", "P2": "S6000", "P3": "N6000", "P4": "F3000/S3000",
        "P5": "F3000/N3000", "P6": "S3000/N3000", "P7": "F2000/S2000/N2000"}


def sh(*a):
    print("$", " ".join(a), flush=True)
    r = subprocess.run(a, cwd=str(BASE), capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True); return r.returncode


def ms(v):
    v = [x for x in v if x is not None]
    if not v: return (None, None)
    return (round(st.mean(v), 3), round(st.stdev(v), 3) if len(v) > 1 else 0.0)


def section(d):
    seeds = ["0", "1", "2"]
    lines = [
        "## 혼합물설계 파일럿 (Phase A, N=6,000) (2026-07-14)",
        "",
        "> 세미나 피드백 대응 재설계(`00_실험재설계_혼합물설계_v1.md`). **총량 N=6,000 완전 고정**, 화염/연기/정상 비율만 변화(simplex-centroid 7셀).",
        "> 순수 단일라벨 풀만(동시라벨 전량 제외). 클립단위 + 시드추출. 시드 0/1/2가 표본·초기화 동시결정, 셀당 n=3.",
        "> ★ **추출 결정(분석세션 판단, 코디네이터 검토요망)**: 재카운트 결과 fire_only 풀이 **97.3% 실내**(실외 763장뿐) → 성분별 inout 층화는 셀마다 도메인이 달라져 §3.4가 막으려던 교란 재현. 따라서 **실내 전용(indoor-only)으로 통일**해 전 셀 도메인을 동일화(순수 성분효과 분리). 실외 일반화는 별도 축.",
        "> 학습: YOLO11n, AdamW lr0 0.001, batch64, **고정 20ep(조기종료 비활성)** — 셀간 유효 학습량(gradient step) 동일화(N고정 통제 정합, 실측 best 출현 ~10~15ep 여유포함). 기존 관례(100ep/pat30)와 다른 HP임 명시.",
        "> P/R은 **고정 conf=0.25**에서 산출(기본 F1-최대 임계값은 런마다 달라 셀간 비교 불가).",
        "> **평가**: 주 지표 = **실내 val 10,512**(실내전용 학습과 정합). 참고 = 전체 val 19,080. 학습중 조기종료 모니터는 실내 val에서 뽑은 **고정 2,000장(클립단위,seed0, 전셀·전시드 공통)** — 최종 지표는 별도 재평가.",
        "> ⚠ V1~V3 검증점 미실행(전향적 검증 프로토콜 — 공식 적합 후 총괄세션 별도 지시). 추출 스크립트: `build_mixture_pools.py`+`run_mixture_pilot.py`.",
        "",
        "### 결과 (mean±std, n=3)",
        "| 셀 | 구성(x_FL,x_SM,x_NM) | 장수 | **실내 mAP@0.5(주)** | fire AP | smoke AP | P | R | 전체 mAP(참고) | held-out |",
        "|----|------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for cell in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        if cell not in d:
            continue
        r = [d[cell][s] for s in seeds if s in d[cell]]
        mp = ms([x["val_in"].get("map50") for x in r])
        fa = ms([x["val_in"].get("fire_ap50") for x in r])
        sa = ms([x["val_in"].get("smoke_ap50") for x in r])
        pp = ms([x["val_in"].get("precision_c25") for x in r])
        rr = ms([x["val_in"].get("recall_c25") for x in r])
        vf = ms([x.get("val_full", {}).get("map50") for x in r])
        ho = ms([x.get("heldout", {}).get("map50") for x in r])
        el = ms([x.get("elapsed_sec") for x in r])
        x = COMP[cell]
        def f(t): return f"{t[0]:.3f}±{t[1]:.3f}" if t[0] is not None else "—"
        note = " *(정상단독,정의상0)*" if cell == "P3" else ""
        lines.append(f"| {cell}{note} | ({x[0]:.2f},{x[1]:.2f},{x[2]:.2f}) | {NIMG[cell]} | **{f(mp)}** | {f(fa)} | {f(sa)} | {f(pp)} | {f(rr)} | {f(vf)} | {f(ho)} |")
    # 학습시간(Phase B 산정용)
    allt = [d[c][s].get("elapsed_sec", 0) for c in d for s in d[c] if d[c][s].get("elapsed_sec")]
    avgt = round(st.mean(allt) / 60, 1) if allt else None
    lines += [
        "",
        "### 메모",
        f"- 셀당 평균 학습시간 ~{avgt}분 (N=6,000, val 19,080 매epoch 포함) — Phase B(N=28,374) 규모 산정 참고.",
        "- 공식 적합(Scheffé special cubic)·V1~V3 예측검증은 **총괄세션이 P1~P7 확인 후 별도 수행**.",
        "- 기존 C1~C4(동시라벨 포함 자연구성)는 현실조건 대조군으로 병기(설계문서 §3.3).",
        "- 산출물: `runs/mixture_pilot.json`, `runs/mixture_pool_index.json`, `compositions/mixture/`.",
        "", "---", ""]
    return "\n".join(lines)


def main():
    if not J.exists():
        print("[중단] mixture_pilot.json 없음."); return 1
    d = json.loads(J.read_text(encoding="utf-8"))
    if not all(len(d.get(c, {})) >= 3 for c in COMP):
        print(f"[중단] 7셀×3시드 미완료: {[(c, len(d.get(c, {}))) for c in COMP]}"); return 1
    txt = LOG.read_text(encoding="utf-8")
    if "## 혼합물설계 파일럿 (Phase A" not in txt:
        txt = txt.replace(ANCHOR, section(d) + ANCHOR, 1)
        LOG.write_text(txt, encoding="utf-8")
        print("[기록] TRAINING_LOG.md에 Phase A 섹션 추가.", flush=True)
    else:
        print("[스킵] 이미 존재.", flush=True)
    sh("git", "pull", "--rebase", "origin", "yhh")
    sh("git", "add", "02_data_ssot/TRAINING_LOG.md", "03_scripts/build_mixture_pools.py",
       "03_scripts/run_mixture_pilot.py", "03_scripts/finalize_mixture.py", "03_scripts/mixture_all.py",
       "compositions/mixture/P1.yaml", "compositions/mixture/P7.yaml")
    if sh("git", "commit", "-m", "혼합물설계 파일럿 Phase A(N=6,000, 7셀×3시드) 결과 기록") == 0:
        sh("git", "push", "origin", "yhh")
    sh("schtasks", "/Delete", "/TN", "DesignA_MIX", "/F")
    print("[완료] Phase A 마무리.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
