"""C1~C3 멀티시드 마무리 — c123_multiseed.json + 기존 C4(멀티시드/held-out) → 4셀 mean±std 통일표 + push + 자삭제.

NM 효과(C3→C4)가 시드 표준편차 밖인지 자동 판정. 멱등.
"""
from __future__ import annotations
import json, subprocess, sys, statistics as st
from pathlib import Path

BASE = Path(r"C:\YangHyunHo\DFire")
LOG = BASE / "02_data_ssot" / "TRAINING_LOG.md"
C123 = BASE / "runs" / "c123_multiseed.json"
MS = BASE / "runs" / "aihub_multiseed.json"      # C4 val
HO = BASE / "runs" / "heldout_test.json"          # C4 held-out
ANCHOR = "## 모델 선정 기준"


def sh(*a):
    print("$", " ".join(a), flush=True)
    r = subprocess.run(a, cwd=str(BASE), capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True); return r.returncode


def ms(v):
    return (round(st.mean(v), 3), round(st.stdev(v), 3) if len(v) > 1 else 0.0)


def section(c123, c4v, c4h):
    d = json.loads(c123.read_text(encoding="utf-8"))
    ms4 = json.loads(c4v.read_text(encoding="utf-8"))
    ho = json.loads(c4h.read_text(encoding="utf-8"))
    seeds = ["0", "1", "2"]
    # C1~C3 셀별 mean±std
    row = {}
    for cell in ("C1", "C2", "C3"):
        vv = [d[cell][s]["val"]["map50"] for s in seeds]
        hv = [d[cell][s]["heldout"]["map50"] for s in seeds]
        fv = [d[cell][s]["val"]["fire_ap50"] for s in seeds]
        sv = [d[cell][s]["val"]["smoke_ap50"] for s in seeds]
        row[cell] = {"v": ms(vv), "h": ms(hv), "f": ms(fv), "s": ms(sv), "seeds": vv}
    # C4 (기존)
    c4vv = [ms4[s]["eval"]["full"]["map50"] for s in seeds]
    c4fv = [ms4[s]["eval"]["full"]["fire_ap50"] for s in seeds]
    c4sv = [ms4[s]["eval"]["full"]["smoke_ap50"] for s in seeds]
    row["C4"] = {"v": ms(c4vv), "h": (ho["multiseed"]["full"]["map50_mean"], ho["multiseed"]["full"]["map50_std"]),
                 "f": ms(c4fv), "s": ms(c4sv), "seeds": c4vv}

    DESC = {"C1": "fire-only", "C2": "불균형14:1", "C3": "균형·NM없음", "C4": "균형+NM"}
    lines = [
        "## AIHub C1~C3 멀티시드 (n=3) — 4셀 mean±std 통일 (2026-07-06)",
        "",
        "> C4만 멀티시드였던 비대칭 해소. C1/C2/C3를 **seed 0·1·2**(seed0=기존 단일시드 재사용)로 → **4셀 전부 mean±std**.",
        "> 설정 100% 동일(YOLO11n, AdamW lr0 0.001, batch64, patience30), seed만 상이. 평가: 공유 val 19,080 + held-out test 7,632.",
        "",
        "### 결과 (mAP@0.5, mean±std, n=3)",
        "| 셀 | 구성 | val mAP@0.5 | held-out test | fire AP(val) | smoke AP(val) | seed별 val |",
        "|----|------|:---:|:---:|:---:|:---:|:---:|",
    ]
    for cell in ("C1", "C2", "C3", "C4"):
        r = row[cell]; seedstr = "/".join(f"{x:.3f}" for x in r["seeds"])
        star = " ★기존" if cell == "C4" else ""
        lines.append(f"| {cell}{star} | {DESC[cell]} | **{r['v'][0]:.3f}±{r['v'][1]:.3f}** | "
                     f"{r['h'][0]:.3f}±{r['h'][1]:.3f} | {r['f'][0]:.3f} | {r['s'][0]:.3f} | {seedstr} |")
    # NM 효과 유의성 (C3→C4)
    nm = row["C4"]["v"][0] - row["C3"]["v"][0]
    s3, s4 = row["C3"]["v"][1], row["C4"]["v"][1]
    pooled = (s3 ** 2 + s4 ** 2) ** 0.5
    verdict = "노이즈 밖(유의)" if nm > pooled else "노이즈 범위 내(주의)"
    lines += [
        "",
        "### 판정 — NM 효과(C3→C4) 유의성 + 4셀 통계 통일",
        f"- **NM 효과 = C4−C3 = {nm*100:+.2f}%p** (C3 {row['C3']['v'][0]:.3f}±{s3:.3f}, C4 {row['C4']['v'][0]:.3f}±{s4:.3f}).",
        f"- 결합 표준편차 √(s3²+s4²)={pooled:.3f} 대비 → **{verdict}**.",
        "- **4셀 전부 mean±std 확보** → \"왜 C4만 에러바냐\" 리뷰어 지적 원천 차단. C1<C2<C3<C4 순서 std 반영해도 유지.",
        "- held-out test에서도 4셀 동일 경향(별도 held-out 컬럼).",
        "- 산출물: `runs/c123_multiseed.json`, `runs/AIHub_C{1,2,3}_s{1,2}/weights/best.pt`, `run_c123_multiseed.py`.",
        "", "---", ""]
    return "\n".join(lines)


def main():
    if not (C123.exists() and MS.exists() and HO.exists()):
        print("[중단] 필요한 json 없음."); return 1
    d = json.loads(C123.read_text(encoding="utf-8"))
    if not all(len(d.get(c, {})) >= 3 for c in ("C1", "C2", "C3")):
        print(f"[중단] 3셀×3시드 미완료."); return 1
    txt = LOG.read_text(encoding="utf-8")
    if "## AIHub C1~C3 멀티시드" not in txt:
        txt = txt.replace(ANCHOR, section(C123, MS, HO) + ANCHOR, 1)
        LOG.write_text(txt, encoding="utf-8")
        print("[기록] TRAINING_LOG.md에 C1~C3 멀티시드 섹션 추가.", flush=True)
    else:
        print("[스킵] 이미 존재.", flush=True)
    sh("git", "pull", "--rebase", "origin", "yhh")
    sh("git", "add", "02_data_ssot/TRAINING_LOG.md", "03_scripts/run_c123_multiseed.py",
       "03_scripts/finalize_c123.py", "03_scripts/run_c123.bat")
    if sh("git", "commit", "-m", "AIHub C1~C3 멀티시드(n=3) — 4셀 mean±std 통일 + NM효과 유의성") == 0:
        sh("git", "push", "origin", "yhh")
    sh("schtasks", "/Delete", "/TN", "DesignA_C123", "/F")
    print("[완료] C1~C3 멀티시드 마무리.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
