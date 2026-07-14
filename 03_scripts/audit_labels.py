"""B2 — AIHub 실내 풀 라벨 무결성 감사 (순수 파싱, GPU 불필요).

교수님 GIGO 지적 방어 근거: 실내 순수 풀(fire_only+smoke_only) 전체 라벨을 스캔해
퇴화/경계밖/중복/극단종횡비 박스를 카운트 → "이상 라벨 X%" 실측치.
YOLO 라벨: class cx cy w h (정규화). 출력: runs/label_audit.json + 콘솔.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

BASE = Path(r"C:\YangHyunHo\DFire")
POOL = BASE / "runs" / "mixture_pool_index.json"
TINY = 0.002          # 극소 박스(정규화 폭·높이 < 0.2%)
AR_EXTREME = 20.0     # 극단 종횡비


def img_to_label(p):
    return Path(p.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")


def main():
    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"]
    # 실내 풀: fire_only + smoke_only (nm은 박스 없음)
    imgs = []
    for comp in ("fire_only", "smoke_only"):
        for clip, fs in pool.get(comp, {}).get("in", {}).items():
            imgs.extend(fs)
    print(f"실내 풀 라벨 감사 대상: {len(imgs):,} 파일", flush=True)

    n_files = 0; n_boxes = 0
    bad = Counter()             # 이상 유형별 박스 수
    files_with_bad = set()
    for ip in imgs:
        lp = img_to_label(ip)
        try:
            lines = [l for l in lp.read_text().splitlines() if l.strip()]
        except Exception:
            continue
        n_files += 1
        seen = set()
        for l in lines:
            parts = l.split()
            if len(parts) < 5:
                bad["malformed"] += 1; files_with_bad.add(ip); continue
            try:
                c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            except Exception:
                bad["malformed"] += 1; files_with_bad.add(ip); continue
            n_boxes += 1
            isbad = False
            if w <= 0 or h <= 0:
                bad["degenerate_zero"] += 1; isbad = True
            elif w < TINY or h < TINY:
                bad["degenerate_tiny"] += 1; isbad = True
            if not (0 <= cx <= 1 and 0 <= cy <= 1) or (cx - w/2 < -1e-6) or (cx + w/2 > 1+1e-6) or (cy - h/2 < -1e-6) or (cy + h/2 > 1+1e-6):
                bad["out_of_bounds"] += 1; isbad = True
            key = (c, round(cx, 5), round(cy, 5), round(w, 5), round(h, 5))
            if key in seen:
                bad["duplicate"] += 1; isbad = True
            seen.add(key)
            if w > 0 and h > 0:
                ar = max(w/h, h/w)
                if ar > AR_EXTREME:
                    bad["extreme_aspect"] += 1; isbad = True
            if isbad:
                files_with_bad.add(ip)

    total_bad = sum(bad.values())
    out = {"files": n_files, "boxes": n_boxes, "anomalies": dict(bad),
           "total_anomalous_boxes": total_bad,
           "pct_boxes_anomalous": round(total_bad / n_boxes * 100, 3) if n_boxes else 0,
           "files_with_any_anomaly": len(files_with_bad),
           "pct_files_anomalous": round(len(files_with_bad) / n_files * 100, 3) if n_files else 0}
    (BASE / "runs" / "label_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== 라벨 무결성 감사 결과 (실내 순수 풀) ===")
    print(f"  파일 {n_files:,} / 박스 {n_boxes:,}")
    for k, v in bad.most_common():
        print(f"  {k}: {v}  ({v/n_boxes*100:.3f}% of boxes)")
    print(f"  이상 박스 총 {total_bad} ({out['pct_boxes_anomalous']}%) / 이상 파일 {len(files_with_bad)} ({out['pct_files_anomalous']}%)")
    print("[저장] runs/label_audit.json")


if __name__ == "__main__":
    main()
