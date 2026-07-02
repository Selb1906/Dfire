"""Design A 마무리 — runs/inout_summary.json → TRAINING_LOG.md 기록 + git push(yhh).

detached 실행 전제(세션과 무관): run_inout.py 완료 후 자동 호출.
멱등: 이미 'Design A' 섹션이 있으면 재기록 스킵. 숫자는 summary.json에서만 인용(SSOT).
판정 문구는 실제 out/in/naive 수치로 자동 생성(가설 성립 여부 무관하게 정확).
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

BASE = Path(r"C:\YangHyunHo\DFire")
LOG = BASE / "02_data_ssot" / "TRAINING_LOG.md"
SUMMARY = BASE / "runs" / "inout_summary.json"
ANCHOR = "## 모델 선정 기준"
D_FULL_11S = 0.787  # 참고 기준(별개 최종 모델, ablation 밖)

NAMES = {"Combined_out_11s": "실외", "Combined_in_11s": "실내", "Combined_naive_11s": "혼합"}


def sh(*args):
    print("$", " ".join(args), flush=True)
    r = subprocess.run(args, cwd=str(BASE), capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True)
    return r.returncode


def row(name, comp, m):
    return (f"| {name} | {comp} | {m['map50']:.3f} | {m['map50_95']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['fire_ap50']:.3f} | {m['smoke_ap50']:.3f} |")


def verdict(out, inn, naive):
    L = []
    o, i, n = out["map50"], inn["map50"], naive["map50"]
    # 1) 실외 vs D_full
    if o >= D_FULL_11S:
        L.append(f"- **실외 결합이 D_full(0.787) 이상**({o:.3f}) — 도메인 일치 데이터 추가가 소폭 도움. 가설 1 성립.")
    else:
        L.append(f"- **실외 결합조차 D_full(0.787) 아래**({o:.3f}, Δ{o-D_FULL_11S:+.3f}) — "
                 f"AIHub는 도메인이 일치(실외)해도 D-Fire test에 도움이 안 됨. **순수 D-Fire가 최선**. 가설 1(실외>D_full) 기각.")
    # 2) 실내 vs 실외 (핵심 신호)
    d = o - i
    if d >= 0.02:
        L.append(f"- **실내가 실외보다 {d*100:.1f}%p 낮음**(in {i:.3f} < out {o:.3f}) — "
                 f"실내 데이터가 특히 D-Fire(실외) 성능을 끌어내림. **A-1 도메인 갭(0.913→0.262)의 주원인=실내 편중**으로 설명 가능. 가설 2 성립.")
    elif d <= -0.02:
        L.append(f"- **오히려 실내가 실외보다 {-d*100:.1f}%p 높음**(in {i:.3f} > out {o:.3f}) — inout 가설과 반대. "
                 f"도메인 갭은 실내/실외 구분보다 다른 요인(스타일·라벨·해상도)에서 기인.")
    else:
        L.append(f"- **실내/실외 차이 미미**(in {i:.3f} vs out {o:.3f}, Δ{d*100:+.1f}%p) — "
                 f"inout은 지배 요인이 아님. 도메인 갭은 데이터셋 스타일 차이 자체가 원인.")
    # 3) naive(혼합) 위치
    L.append(f"- 혼합(naive) {n:.3f} — 실외 {o:.3f}/실내 {i:.3f} 사이 " +
             ("(예상대로 중간)." if min(o, i) - 0.005 <= n <= max(o, i) + 0.005 else "밖(추가 요인 시사)."))
    # 4) 종합
    if o < D_FULL_11S and i < D_FULL_11S and n < D_FULL_11S:
        L.append("- **종합**: 어떤 inout 서브셋도 D_full을 넘지 못함 → 결론 \"AIHub 결합은 D-Fire 벤치마크에 무익, "
                 "규모·도메인 확장보다 대상 도메인 데이터가 지배적\". §6.3 도메인 갭 논거 강화.")
    return "\n".join(L)


def build_section(res):
    m = {r["name"]: r["test"] for r in res}
    out, inn, naive = m["Combined_out_11s"], m["Combined_in_11s"], m["Combined_naive_11s"]
    s = []
    s.append("## Design A — AIHub 실내/실외 서브셋 결합 통제실험 (2026-07-02, 도메인 갭 원인 규명) ★ablation과 별개")
    s.append("")
    s.append("> 목적: A-1 도메인 갭(AIHub→DFire 0.913→0.262)의 원인이 \"실내 편중\"인지 검증.")
    s.append("> 통제: 모든 셀 = D-Fire train ×3(42,366) + AIHub 14,122(클래스 SM:FL:NONE=1:1:1 매칭). **inout만 변수**.")
    s.append("> 모델 YOLO11s(D_full 11s=0.787과 직접 비교), R8 동일 하이퍼파라미터. 평가 = D-Fire test 4,306.")
    s.append("> D-Fire 육안 표본조사: test 이미지 주로 실외(AoF 감시·WEB 산불/뉴스) → \"D-Fire≈실외\" 전제.")
    s.append("")
    s.append("### 결과 (DFire test 4,306, mAP@0.5)")
    s.append("| 셀 | 구성(AIHub 부분) | mAP@0.5 | mAP@0.5:0.95 | P | R | fire AP | smoke AP |")
    s.append("|----|------|:---:|:---:|:---:|:---:|:---:|:---:|")
    s.append(row("Combined_out", "실외 14,122", out))
    s.append(row("Combined_in", "실내 14,122", inn))
    s.append(row("Combined_naive", "혼합 14,122", naive))
    s.append("| (참고) **D_full 11s** | AIHub 없음·DFire 전체 | **0.787** | 0.458 | 0.783 | 0.719 | 0.729 | 0.845 |")
    s.append("| (참고) AIHub C4→DFire | AIHub 단독 114K | 0.262 | 0.130 | 0.423 | 0.319 | 0.127 | 0.397 |")
    s.append("")
    s.append("### 판정")
    s.append(verdict(out, inn, naive))
    s.append("- 산출물: `runs/Combined_{out,in,naive}_11s/weights/best.pt`, 집계 `runs/inout_summary.json`, 구성 `build_inout_subsets.py`.")
    s.append("")
    s.append("---")
    s.append("")
    return "\n".join(s)


def main():
    res = json.loads(SUMMARY.read_text(encoding="utf-8"))
    need = set(NAMES)
    have = {r["name"] for r in res}
    if not need <= have:
        print(f"[중단] 3셀 미완료: 없음={need-have}. 기록 스킵.", flush=True)
        return 1
    txt = LOG.read_text(encoding="utf-8")
    if "Design A —" in txt:
        print("[스킵] Design A 섹션 이미 존재.", flush=True)
    else:
        section = build_section(res)
        txt = txt.replace(ANCHOR, section + ANCHOR, 1)
        txt = txt.replace(
            "*마지막 업데이트: 2026-07-02 (D_full 0.789 헤드라인 + dedup 강건성: 누수영향 1~3%p, ablation 견고)*",
            "*마지막 업데이트: 2026-07-02 (Design A 실내/실외 결합 통제실험 기록 — 도메인 갭 원인 규명)*")
        LOG.write_text(txt, encoding="utf-8")
        print("[기록] TRAINING_LOG.md에 Design A 섹션 추가.", flush=True)

    # git: pull --rebase → add → commit(yhh) → push (best-effort, 로그만)
    sh("git", "pull", "--rebase", "origin", "yhh")
    sh("git", "add", "02_data_ssot/TRAINING_LOG.md", "03_scripts/finalize_inout.py",
       "03_scripts/run_inout.py", "03_scripts/build_inout_subsets.py",
       "compositions/combined_out.yaml", "compositions/combined_in.yaml")
    rc = sh("git", "commit", "-m",
            "Design A(실내/실외) 결합 통제실험 결과 기록 + detached 마무리 스크립트")
    if rc == 0:
        sh("git", "push", "origin", "yhh")
    else:
        print("[정보] 커밋할 변경 없음(또는 이미 기록됨).", flush=True)
    print("[완료] Design A 마무리.", flush=True)
    # Phase2(X 크로스평가 + T1 전이학습) 자동 체이닝 — GPU 유휴 상태에서 이어서 실행.
    try:
        rc = sh("schtasks", "/Run", "/TN", "DesignA_Phase2")
        print(f"[체이닝] DesignA_Phase2 트리거 (rc={rc}).", flush=True)
    except Exception as e:
        print(f"[경고] Phase2 트리거 실패: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
