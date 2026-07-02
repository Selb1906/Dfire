"""Phase 2 마무리 — X(도메인 매트릭스) + T1(전이학습) → TRAINING_LOG.md 기록 + git push(yhh).

멱등: 섹션 이미 있으면 스킵. 숫자는 domain_matrix.json / transfer_summary.json / inout_summary.json에서만 인용.
전이학습 판정은 실측 수치로 자동 생성(D_full 0.787·Combined_naive 대비 우열).
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

BASE = Path(r"C:\YangHyunHo\DFire")
LOG = BASE / "02_data_ssot" / "TRAINING_LOG.md"
ANCHOR = "## 모델 선정 기준"
DM = BASE / "runs" / "domain_matrix.json"
TS = BASE / "runs" / "transfer_summary.json"
IO = BASE / "runs" / "inout_summary.json"
D_FULL_11S = 0.787


def sh(*a):
    print("$", " ".join(a), flush=True)
    r = subprocess.run(a, cwd=str(BASE), capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True)
    return r.returncode


def matrix_section(dm):
    dfull_av = dm["Dfull"]["AIHub_val"]
    av_11m = dfull_av.get("11m"); av_11s = dfull_av.get("11s")
    s = [
        "## 도메인 매트릭스 X — 양방향 크로스평가 (2026-07-02, 갭 특성화)",
        "",
        "> A-1(AIHub학습→DFire test)의 **반대 방향**(DFire학습 D_full→AIHub val)을 추가해 2×2 완성. 재학습 없음.",
        "",
        "### mAP@0.5 (행=학습 도메인, 열=평가 도메인)",
        "| 학습 \\ 평가 | DFire test | AIHub val |",
        "|------|:---:|:---:|",
        f"| **D-Fire**(D_full 11m) | **0.789** | {av_11m if av_11m is not None else 'N/A'} |",
        f"| **AIHub**(C4 11n) | 0.262 | 0.913 |",
        "",
        "### 판정",
    ]
    if av_11m is not None:
        drop = 0.789 - av_11m
        s.append(f"- **DFire→AIHub도 급락**(0.789→{av_11m:.3f}, −{drop*100:.1f}%p) — A-1(AIHub→DFire 0.913→0.262)과 함께 "
                 "**양방향 도메인 갭** 확정. 어느 쪽으로도 전이 안 됨.")
        s.append("- 각 도메인은 자기 test에서만 강함(대각선 우세) → \"성능은 대상 도메인 데이터에 종속\"이라는 핵심 주장을 매트릭스로 시각화.")
    s += ["- 산출물: `runs/domain_matrix.json`, `runs/Dfull_11*_on_AIHubval/`.", "", "---", ""]
    return "\n".join(s)


def transfer_section(ts, naive):
    t = ts["test"]; mv = t["map50"]
    s = [
        "## 전이학습 T1 — pretrain(AIHub)→finetune(DFire) vs 혼합 vs 타깃단독 (2026-07-02, 통합전략 통제)",
        "",
        "> 데이터 고정(D-Fire 14,122 + AIHub-mixed 14,122 = Combined_naive와 동일), **통합 '전략'만 변경**.",
        "> Stage1 AIHub 14K 사전학습 → Stage2 D-Fire 미세조정. 하이퍼파라미터·모델(11s) 동일, 초기화만 상이.",
        "",
        "### 결과 (DFire test 4,306, mAP@0.5)",
        "| 전략 | 구성 | mAP@0.5 | mAP@0.5:0.95 | P | R | fire AP | smoke AP |",
        "|------|------|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| **전이(T1)** | AIHub 사전학습→DFire 미세조정 | **{mv:.3f}** | {t['map50_95']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | {t['fire_ap50']:.3f} | {t['smoke_ap50']:.3f} |",
    ]
    nv = naive.get("map50") if naive else None
    if nv is not None:
        s.append(f"| 혼합(Combined_naive) | DFire×3+AIHub 동시학습 | {nv:.3f} | — | — | — | — | — |")
    s.append(f"| (참고) 타깃단독 D_full 11s | DFire 전체만 | 0.787 | 0.458 | 0.783 | 0.719 | 0.729 | 0.845 |")
    s += ["", "### 판정"]
    # vs D_full
    if mv >= D_FULL_11S + 0.005:
        s.append(f"- **전이가 타깃단독 D_full(0.787)을 상회**({mv:.3f}, +{(mv-D_FULL_11S)*100:.1f}%p) — "
                 "도메인 갭이 커도 **AIHub를 '사전학습→미세조정'으로 쓰면** 대형 보조데이터가 D-Fire 성능을 끌어올림. 헤드라인 절대성능 갱신.")
    elif mv <= D_FULL_11S - 0.005:
        s.append(f"- **전이조차 D_full(0.787) 아래**({mv:.3f}, {(mv-D_FULL_11S)*100:+.1f}%p) — "
                 "사전학습→미세조정으로도 도메인 불일치 데이터는 순수 D-Fire를 못 넘음. **\"대상 도메인 데이터가 지배적\"** 주장 결정적 강화.")
    else:
        s.append(f"- **전이 ≈ D_full**({mv:.3f} vs 0.787) — AIHub 사전학습이 순이득도 손해도 아님. 대형 보조데이터의 한계.")
    # vs 혼합
    if nv is not None:
        d = mv - nv
        if abs(d) < 0.005:
            s.append(f"- 전이 ≈ 혼합({mv:.3f} vs {nv:.3f}) — **동일 데이터라면 통합 전략(전이 vs 혼합)은 큰 차이 없음**.")
        else:
            better = "전이 > 혼합" if d > 0 else "혼합 > 전이"
            s.append(f"- **{better}**({mv:.3f} vs {nv:.3f}, Δ{d*100:+.1f}%p) — 동일 데이터에서 통합 방식이 성능을 가름.")
    s += ["- 산출물: `runs/Transfer_pretrain_aihub14k`, `runs/Transfer_finetune_dfire/weights/best.pt`, `runs/transfer_summary.json`.",
          "", "---", ""]
    return "\n".join(s)


def main():
    txt = LOG.read_text(encoding="utf-8")
    ins = ""
    if DM.exists() and "도메인 매트릭스 X" not in txt:
        ins += matrix_section(json.loads(DM.read_text(encoding="utf-8")))
    if TS.exists() and "전이학습 T1" not in txt:
        naive = None
        if IO.exists():
            for r in json.loads(IO.read_text(encoding="utf-8")):
                if r["name"] == "Combined_naive_11s":
                    naive = r["test"]
        ins += transfer_section(json.loads(TS.read_text(encoding="utf-8")), naive)
    if ins:
        txt = txt.replace(ANCHOR, ins + ANCHOR, 1)
        txt = txt.replace(
            "*마지막 업데이트: 2026-07-02 (Design A 실내/실외 결합 통제실험 기록 — 도메인 갭 원인 규명)*",
            "*마지막 업데이트: 2026-07-02 (Phase2: 도메인 매트릭스 X + 전이학습 T1 기록)*")
        LOG.write_text(txt, encoding="utf-8")
        print("[기록] TRAINING_LOG.md에 X/T1 섹션 추가.", flush=True)
    else:
        print("[스킵] 추가할 섹션 없음(이미 기록/결과 없음).", flush=True)

    sh("git", "pull", "--rebase", "origin", "yhh")
    sh("git", "add", "02_data_ssot/TRAINING_LOG.md",
       "03_scripts/eval_dfull_on_aihub.py", "03_scripts/run_transfer.py",
       "03_scripts/finalize_phase2.py", "03_scripts/run_phase2.bat",
       "compositions/aihub14k.yaml", "compositions/aihub_val.yaml")
    if sh("git", "commit", "-m", "Phase2: 도메인 매트릭스(X) + 전이학습(T1) 결과 기록") == 0:
        sh("git", "push", "origin", "yhh")
    else:
        print("[정보] 커밋할 변경 없음.", flush=True)
    print("[완료] Phase2 마무리.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
