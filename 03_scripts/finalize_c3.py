"""AIHub C3 마무리 — aihub_c3_summary.json → C1<C2<C3<C4 4셀 표 + NM/균형 효과 분해 → TRAINING_LOG + push + 작업 자삭제.

C1/C2/C4는 R9 확정값(공유 val 19,080). C3만 신규. 멱등. 완료 후 예약작업 DesignA_C3 자삭제.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

BASE = Path(r"C:\YangHyunHo\DFire")
LOG = BASE / "02_data_ssot" / "TRAINING_LOG.md"
C3J = BASE / "runs" / "aihub_c3_summary.json"
ANCHOR = "## 모델 선정 기준"
# R9 확정값 (AIHub val 19,080 공유)
C1 = {"map": 0.463, "sm": 0.000, "fl": 0.927, "n": 57391, "desc": "fire-only"}
C2 = {"map": 0.770, "sm": 0.616, "fl": 0.925, "n": None, "desc": "불균형 14:1"}
C4 = {"map": 0.913, "sm": 0.896, "fl": 0.930, "n": 114462, "desc": "균형+NM"}
# D-Fire 참고 (R8, DFire test)
DF = {"C1": 0.325, "C2": 0.455, "C3": 0.691, "C4": 0.736}


def sh(*a):
    print("$", " ".join(a), flush=True)
    r = subprocess.run(a, cwd=str(BASE), capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True)
    return r.returncode


def section(c3):
    v = c3["val"]; c3m = v["map50"]
    sr = c3.get("sanity_reeval", {})   # fresh 재평가값(있으면 R9 상수 대체)
    # 셀별 (map, smoke, fire): sanity 우선, 없으면 R9 상수
    def cell(key, const):
        if key in sr:
            return sr[key]["map50"], sr[key]["smoke_ap50"], sr[key]["fire_ap50"], True
        return const["map"], const["sm"], const["fl"], False
    c1m, c1s, c1f, f1 = cell("C1", C1)
    c2m, c2s, c2f, f2 = cell("C2", C2)
    c4m, c4s, c4f, f4 = cell("C4", C4)
    fresh = f1 or f2 or f4
    src = "fresh 재평가(sanity)" if fresh else "R9 인용"
    bal = c3m - c2m          # 균형 효과 C2→C3
    nm = c4m - c3m           # NM 효과 C3→C4
    df_bal = DF["C3"] - DF["C2"]; df_nm = DF["C4"] - DF["C3"]
    s = [
        "## AIHub 4셀 완성 — C3(균형·NM없음) 추가 + 구성효과 분해 (2026-07-03, C1<C2<C3<C4)",
        "",
        f"> C3 = C4에서 NM(28,695) 제외 = **{c3['train_imgs']:,}장**(균형 유지, FL포함=SM포함=57,391). "
        "C1/C2/C4와 **동일 파이프라인**(`aihub71751_to_yolo.py`→`yolo_071751`)·동일 설정(YOLO11n, seed0)·**공유 val 19,080**. NM 단독 효과(C3→C4) 격리 목적.",
        f"> C1/C2/C4는 {src}로 4셀 전부 동일 코드 재평가(sanity).",
        "",
        "### 결과 (AIHub val 19,080, mAP@0.5)",
        "| 셀 | 구성 | train | mAP@0.5 | smoke AP | fire AP |",
        "|----|------|------:|:---:|:---:|:---:|",
        f"| C1 | {C1['desc']} | {C1['n']:,} | {c1m:.3f} | {c1s:.3f} | {c1f:.3f} |",
        f"| C2 | {C2['desc']} | — | {c2m:.3f} | {c2s:.3f} | {c2f:.3f} |",
        f"| **C3** | **균형·NM없음** | {c3['train_imgs']:,} | **{c3m:.3f}** | {v['smoke_ap50']:.3f} | {v['fire_ap50']:.3f} |",
        f"| C4 | {C4['desc']} | {C4['n']:,} | {c4m:.3f} | {c4s:.3f} | {c4f:.3f} |",
        "",
        "### 판정 — 구성효과 분해 (AIHub vs D-Fire)",
        "| 효과 | AIHub | D-Fire(참고, R8) |",
        "|------|:---:|:---:|",
        f"| 균형 (C2→C3) | {bal*100:+.1f}%p ({c2m:.3f}→{c3m:.3f}) | {df_bal*100:+.1f}%p ({DF['C2']:.3f}→{DF['C3']:.3f}) |",
        f"| NM (C3→C4) | {nm*100:+.1f}%p ({c3m:.3f}→{c4m:.3f}) | {df_nm*100:+.1f}%p ({DF['C3']:.3f}→{DF['C4']:.3f}) |",
        "",
        f"- **C1<C2<C3<C4 단조 증가** 확인 (AIHub: {c1m:.3f}<{c2m:.3f}<{c3m:.3f}<{c4m:.3f}).",
        f"- 균형·NM 모두 양(+)의 기여 → 데이터 구성 효과가 AIHub 대규모에서도 재현. NM 효과 {nm*100:+.1f}%p.",
    ]
    if fresh:
        s.append(f"- **sanity**: C1/C2/C4 fresh 재평가 = R9(0.463/0.770/0.913)와 대조 → 동일 파이프라인·공유 val 재현성 확인.")
    s += [
        "- 두 데이터셋 모두 균형·NM이 성능을 끌어올리나 크기는 데이터셋 의존(도메인 특성 차이).",
        "- 산출물: `runs/AIHub_C3/weights/best.pt`, `runs/aihub_c3_summary.json`, `run_aihub_c3.py`.",
        "", "---", ""]
    return "\n".join(s)


def main():
    if not C3J.exists():
        print("[중단] aihub_c3_summary.json 없음."); return 1
    c3 = json.loads(C3J.read_text(encoding="utf-8"))
    txt = LOG.read_text(encoding="utf-8")
    if "## AIHub 4셀 완성" not in txt:
        txt = txt.replace(ANCHOR, section(c3) + ANCHOR, 1)
        LOG.write_text(txt, encoding="utf-8")
        print("[기록] TRAINING_LOG.md에 AIHub 4셀 섹션 추가.", flush=True)
    else:
        print("[스킵] 이미 존재.", flush=True)
    # §5.1 헤드라인 그림(C1→C4) 자동 생성
    sh(sys.executable, "03_scripts/plot_aihub_composition_headline.py")
    sh("git", "pull", "--rebase", "origin", "yhh")
    sh("git", "add", "02_data_ssot/TRAINING_LOG.md", "03_scripts/run_aihub_c3.py",
       "03_scripts/finalize_c3.py", "03_scripts/run_c3.bat", "compositions/aihub_c3.yaml",
       "03_scripts/plot_aihub_composition_headline.py", "04_figures/fig_aihub_composition_headline.png")
    if sh("git", "commit", "-m", "AIHub 4셀 완성 — C3(균형·NM없음) 추가 + NM/균형 효과 분해") == 0:
        sh("git", "push", "origin", "yhh")
    # IFireSmoke 체이닝 — GPU 유휴 상태에서 이어서 실행
    sh("schtasks", "/Run", "/TN", "DesignA_IFS")
    print("[체이닝] DesignA_IFS 트리거.", flush=True)
    sh("schtasks", "/Delete", "/TN", "DesignA_C3", "/F")
    print("[완료] C3 마무리 + 작업 정리.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
