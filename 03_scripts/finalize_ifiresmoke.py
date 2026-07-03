"""IFireSmoke 마무리 — ifiresmoke_summary.json → TRAINING_LOG §5.9 직접비교 섹션 + push + 작업 자삭제."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

BASE = Path(r"C:\YangHyunHo\DFire")
LOG = BASE / "02_data_ssot" / "TRAINING_LOG.md"
SUM = BASE / "runs" / "ifiresmoke_summary.json"
ANCHOR = "## 모델 선정 기준"


def sh(*a):
    print("$", " ".join(a), flush=True)
    r = subprocess.run(a, cwd=str(BASE), capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True)
    return r.returncode


def section(d):
    t = d["test"]; v = d["val"]
    s = [
        "## IFireSmoke 네이티브 직접 벤치 — §5.9 '인용→직접비교' (2026-07-03)",
        "",
        "> Sozol et al.(PLOS ONE PONE-D-24-47312) 공개 IFireSmoke(HF `shahriar-5/IFireSmoke`, 5,000장)에 우리 파이프라인 직접 적용.",
        "> ⚠️ **이 데이터셋은 이미 FL:SM≈1:1 균형 + NM 0장** → 우리 '구성방법(균형+NM)'은 적용 불가/무의미.",
        ">   따라서 '같은 데이터 위 우리 모델(YOLO11n)' **직접비교**로 수행(그들 Enhanced-YOLOv5 보고치 대비).",
        "> train 4000(native) → **test 500** 평가. 클래스 0=Fire, 1=Smoke(원본 순서 유지).",
        "",
        "### 결과 (IFireSmoke test 500, mAP@0.5)",
        "| 모델 | mAP@0.5 | mAP@0.5:0.95 | fire AP | smoke AP | P | R |",
        "|------|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| **우리 YOLO11n (native)** | **{t['map50']:.3f}** | {t['map50_95']:.3f} | {t['fire_ap50']:.3f} | {t['smoke_ap50']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} |",
        "| (참고) 그들 Enhanced-YOLOv5 | `[보고치: 논문 인용 — 논문세션 확인]` | — | — | — | — | — |",
        f"| (참고) 우리 valid 500 | {v['map50']:.3f} | {v['map50_95']:.3f} | {v['fire_ap50']:.3f} | {v['smoke_ap50']:.3f} | — | — |",
        "",
        "### 메모",
        f"- 우리 YOLO11n이 IFireSmoke test에서 **mAP@0.5 {t['map50']:.3f}** 달성 → §5.9를 '인용'에서 '같은 데이터 직접 적용'으로 강화.",
        "- **한계 명시**: 이 데이터는 NM 0장·이미 균형이라 우리 핵심 기여(NM 비화재보 저감)를 실증하는 셋이 아님. NM 효과 실증은 별도 외부 NM 주입 실험 필요(후속 옵션).",
        "- 그들 보고치는 논문세션이 원문에서 인용 확정(분석세션 범위 밖).",
        "- 산출물: `runs/IFireSmoke_11n/weights/best.pt`, `runs/ifiresmoke_summary.json`, `run_ifiresmoke.py`.",
        "", "---", ""]
    return "\n".join(s)


def main():
    if not SUM.exists():
        print("[중단] ifiresmoke_summary.json 없음."); return 1
    d = json.loads(SUM.read_text(encoding="utf-8"))
    txt = LOG.read_text(encoding="utf-8")
    if "## IFireSmoke 네이티브 직접 벤치" not in txt:
        txt = txt.replace(ANCHOR, section(d) + ANCHOR, 1)
        LOG.write_text(txt, encoding="utf-8")
        print("[기록] TRAINING_LOG.md에 IFireSmoke 섹션 추가.", flush=True)
    else:
        print("[스킵] 이미 존재.", flush=True)
    sh("git", "pull", "--rebase", "origin", "yhh")
    sh("git", "add", "02_data_ssot/TRAINING_LOG.md", "03_scripts/run_ifiresmoke.py",
       "03_scripts/finalize_ifiresmoke.py", "03_scripts/run_ifs.bat", ".gitignore")
    if sh("git", "commit", "-m", "IFireSmoke 네이티브 직접 벤치 — §5.9 직접비교 (NM 0·이미균형: 모델 직접비교)") == 0:
        sh("git", "push", "origin", "yhh")
    sh("schtasks", "/Delete", "/TN", "DesignA_IFS", "/F")
    print("[완료] IFireSmoke 마무리 + 작업 정리.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
