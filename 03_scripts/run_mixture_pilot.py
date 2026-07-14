"""혼합물설계 Phase A 러너 — N=6,000 고정 7셀 × 3시드 = 21회 학습·평가.

시드 하나가 표본추출+학습초기화 동시 결정. inout 층화 + 클립단위 추출. 순수 단일라벨 풀만.
평가: 공유 val 19,080 + held-out 7,632. P3(정상단독)은 학습 에러 시 mAP=0(정의상) 기록.
resumable(json 증분) + done가드. 하이퍼파라미터 기존과 100% 동일(YOLO11n).
"""
from __future__ import annotations
import json, os, time, random
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

BASE = Path(r"C:\YangHyunHo\DFire")
PROJECT = str(BASE / "runs")
COMP = BASE / "compositions" / "mixture"
COMP.mkdir(parents=True, exist_ok=True)
POOL = BASE / "runs" / "mixture_pool_index.json"
MONITOR_LIST = (BASE / "compositions" / "mixture" / "monitor_val_in2k_list.txt").as_posix()  # 조기종료 모니터(실내 2K 고정, 전셀 공통)
VAL_IN = str(BASE / "compositions" / "aihub_val_in.yaml")      # 실내 val 10,512 (주 지표)
VAL_FULL = str(BASE / "compositions" / "aihub_val.yaml")       # 전체 val 19,080 (참고)
HELDOUT = str(BASE / "compositions" / "heldout_full.yaml")     # held-out 7,632 (여유시)
IMGSZ, BATCH, DEVICE, EPOCHS = 640, 64, "0", 20   # 고정 20ep(조기종료 비활성) — 셀간 유효학습량 통제(N고정 정합)

CELLS = {  # name: (fire_only, smoke_only, nm)
    "P1": (6000, 0, 0), "P2": (0, 6000, 0), "P3": (0, 0, 6000),
    "P4": (3000, 3000, 0), "P5": (3000, 0, 3000), "P6": (0, 3000, 3000),
    "P7": (2000, 2000, 2000)}
SEEDS = [0, 1, 2]
HP = dict(epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, project=PROJECT,
          exist_ok=True, save_period=0, patience=999, optimizer="AdamW",   # patience>ep → 조기종료 없음(고정 20ep 완주)
          lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3, cache=False,
          hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
          mosaic=1.0, mixup=0.1, copy_paste=0.0, verbose=False, plots=False)


def sample(pool_comp, k, seed):
    """실내 전용(indoor-only) + 클립단위 + 시드 무작위로 k장 추출.

    근거: fire_only 풀이 97.3% 실내(실외 763장뿐)라 성분별 inout 층화는 셀마다 도메인 구성이
    달라져 설계 §3.4가 막으려던 도메인 교란을 재현함. 실내 전용으로 통일하면 전 셀 도메인 동일
    → 순수 '성분 효과'만 분리(실외 일반화는 별도 축). 실내 풀: fire 27,613/smoke 12,253/nm 14,260 (N=6,000 충분).
    """
    if k <= 0:
        return []
    rng = random.Random(seed * 1000 + 7)
    clips = list(pool_comp.get("in", {}).items())   # 실내 전용
    rng.shuffle(clips)
    buf = []
    for _, imgs in clips:
        ii = list(imgs); rng.shuffle(ii); buf.extend(ii)
        if len(buf) >= k:
            break
    if len(buf) < k:   # 실내 부족 시에만 실외 보충(거의 없음)
        for _, imgs in pool_comp.get("out", {}).items():
            buf.extend(imgs)
    return buf[:k]


def build_yaml(name, imgs):
    lst = COMP / f"{name}_train.txt"
    lst.write_text("\n".join(imgs) + "\n", encoding="utf-8")
    y = COMP / f"{name}.yaml"
    y.write_text(f"# 혼합물설계 {name} ({len(imgs)}장, N고정). 순수 단일라벨. 0=smoke,1=fire.\n"
                 f"# 학습중 val = 실내 2K 모니터(조기종료 전용). 최종지표는 실내10,512/전체19,080 별도 평가.\n"
                 f"train: {lst.as_posix()}\n"
                 f"val: {MONITOR_LIST}\n\nnames: ['smoke', 'fire']\nnc: 2\n",
                 encoding="utf-8")
    return str(y)


def metrics(res):
    ap50 = [float(v) for v in res.box.ap50]
    return {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "smoke_ap50": round(ap50[0], 4) if ap50 else None,
            "fire_ap50": round(ap50[1], 4) if len(ap50) > 1 else None}


def done(name):
    b = Path(PROJECT) / name / "weights" / "best.pt"
    return b.exists() and b.stat().st_size < 10 * 1024 * 1024


def main():
    from ultralytics import YOLO
    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"]
    out = Path(PROJECT) / "mixture_pilot.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for cell, (nf, ns, nn) in CELLS.items():
        res.setdefault(cell, {})
        for seed in SEEDS:
            key = str(seed)
            if key in res[cell] and "val_in" in res[cell][key]:
                print(f"[{cell} s{seed}] 집계됨 — 건너뜀", flush=True); continue
            name = f"MIX_{cell}_s{seed}"
            imgs = (sample(pool.get("fire_only", {}), nf, seed)
                    + sample(pool.get("smoke_only", {}), ns, seed)
                    + sample(pool.get("nm", {}), nn, seed))
            data = build_yaml(name, imgs)
            best = Path(PROJECT) / name / "weights" / "best.pt"
            last = Path(PROJECT) / name / "weights" / "last.pt"
            t0 = time.time()
            train_err = None
            if not done(name):
                try:
                    if last.exists():
                        print(f"[{cell} s{seed}] resume ({len(imgs)}장)", flush=True)
                        YOLO(str(last)).train(resume=True)
                    else:
                        print(f"[{cell} s{seed}] 학습 시작 ({len(imgs)}장 = F{nf}/S{ns}/N{nn})", flush=True)
                        YOLO("yolo11n.pt").train(data=data, name=name, seed=seed, **HP)
                except Exception as e:
                    train_err = str(e)[:200]
                    print(f"[{cell} s{seed}] 학습 에러: {train_err}", flush=True)
            elapsed = round(time.time() - t0, 1)
            rec = {"n_images": len(imgs), "fire": nf, "smoke": ns, "nm": nn, "elapsed_sec": elapsed}
            if best.exists() and not train_err:
                model = YOLO(str(best))
                rec["val_in"] = metrics(model.val(data=VAL_IN, imgsz=IMGSZ, device=DEVICE, split="val",
                                                  project=PROJECT, name=f"{name}_valin", exist_ok=True, plots=False, verbose=False))
                # B1: 셀간 비교 위해 P/R을 고정 conf=0.25에서 재산출(기본 P/R은 런마다 F1최대 임계값이라 비교불가)
                r25 = model.val(data=VAL_IN, imgsz=IMGSZ, device=DEVICE, split="val", conf=0.25,
                                project=PROJECT, name=f"{name}_valin_c25", exist_ok=True, plots=False, verbose=False)
                rec["val_in"]["precision_c25"] = round(float(r25.box.mp), 4)
                rec["val_in"]["recall_c25"] = round(float(r25.box.mr), 4)
                rec["val_full"] = metrics(model.val(data=VAL_FULL, imgsz=IMGSZ, device=DEVICE, split="val",
                                                    project=PROJECT, name=f"{name}_valfull", exist_ok=True, plots=False, verbose=False))
                rec["heldout"] = metrics(model.val(data=HELDOUT, imgsz=IMGSZ, device=DEVICE, split="val",
                                                   project=PROJECT, name=f"{name}_heldout", exist_ok=True, plots=False, verbose=False))
            else:
                rec["val_in"] = {"map50": 0.0, "note": "정상단독/학습에러 → mAP=0(정의상)", "error": train_err}
                rec["val_full"] = {"map50": 0.0}; rec["heldout"] = {"map50": 0.0}
            res[cell][key] = rec
            out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            v = rec["val_in"].get("map50")
            print(f"[{cell} s{seed}] 실내val mAP50={v} (전체 {rec['val_full'].get('map50')})  ({elapsed/60:.1f}분)", flush=True)
    print("\n===== Phase A 요약 (셀 실내val mAP50) =====", flush=True)
    import statistics as st
    for cell in CELLS:
        vs = [res[cell][str(s)]["val_in"]["map50"] for s in SEEDS if str(s) in res[cell]]
        if len(vs) == 3:
            print(f"  {cell}: {st.mean(vs):.4f}±{st.stdev(vs):.4f}", flush=True)


if __name__ == "__main__":
    main()
