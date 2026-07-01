"""결합 학습(오버샘플) 구성 — D-Fire 지배(75%) + AIHub 보충(25%).

배경: A-1에서 AIHub→DFire 도메인 갭(0.913→0.262) 확인. 예전 A-2(89% AIHub 단순병합)는
D_full(0.789)보다 낮을 것이 유력 → D-Fire를 오버샘플링해 지배적으로 재설계.
구성: D-Fire train(14,122) ×3 + AIHub 균등샘플(14,122) = 56,488 (D-Fire 75%).
오버샘플링은 이미지 복제 없이 train 리스트 파일에 D-Fire 경로를 3회 기재.
출력: compositions/combined_os_train.txt + compositions/combined_os.yaml
"""
from __future__ import annotations
from pathlib import Path

DFIRE_TRAIN = Path(r"C:\YangHyunHo\DFire\dfire\data\train\images")
AIHUB_TRAIN = Path(r"D:\AIHub_Fire\yolo_071751\images\train")
OUTDIR = Path(r"C:\YangHyunHo\DFire\compositions")
LIST = OUTDIR / "combined_os_train.txt"
YAML = OUTDIR / "combined_os.yaml"
DFIRE_REPEAT = 3        # D-Fire 오버샘플 배수
IMG_EXT = {".jpg", ".jpeg", ".png"}


def imgs(d):
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def even_sample(items, k):
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def main():
    dfire = imgs(DFIRE_TRAIN)
    aihub_all = imgs(AIHUB_TRAIN)
    aihub = even_sample(aihub_all, len(dfire))   # AIHub를 D-Fire 크기로 균등 샘플
    lines = dfire * DFIRE_REPEAT + aihub          # D-Fire ×3 + AIHub ×1
    LIST.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_df = len(dfire) * DFIRE_REPEAT
    n_ah = len(aihub)
    tot = n_df + n_ah
    YAML.write_text(
        f"# 결합 오버샘플 — D-Fire ×{DFIRE_REPEAT}({n_df}) + AIHub 샘플({n_ah}) = {tot} (D-Fire {n_df/tot*100:.0f}%)\n"
        f"# val/test = D-Fire 원본. 클래스 0=smoke,1=fire (양쪽 통일).\n"
        f"train: {LIST.as_posix()}\n"
        f"val: C:/YangHyunHo/DFire/dfire/data/val/images\n"
        f"test: C:/YangHyunHo/DFire/dfire/data/test/images\n\n"
        f"names: ['smoke', 'fire']\nnc: 2\n", encoding="utf-8")
    print(f"[done] D-Fire {len(dfire)}×{DFIRE_REPEAT}={n_df} + AIHub {n_ah} = {tot} (D-Fire {n_df/tot*100:.0f}%)")
    print(f"  list: {LIST}\n  yaml: {YAML}")


if __name__ == "__main__":
    main()
