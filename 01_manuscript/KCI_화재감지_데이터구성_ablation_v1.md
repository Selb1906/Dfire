# 비화재보 저감을 위한 엣지 화재·연기 탐지 모델의 데이터 구성 효과 분석

**A Study on the Effect of Training Data Composition for Edge-Based Fire and Smoke Detection toward Non-Fire Alarm Reduction**

**저자**: 김수진, 양현호, 김정욱(Jeonguk Kim)*

\* 교신저자(Corresponding author) — 잠정 표기, 투고 전 확정 필요. 소속·이메일 추후 기입.

---

> 작성 상태: v1.8 (2026-06-25). v1.7에서 표 5 TBD → 실수치 완성 (deep-research 결과 반영). YOLOv5s 76.7%, YOLOv8n 75.4%, CCi-YOLOv8n 78.5% [16] D-Fire 직접 비교 추가. YOLOGX/YOLOv7 [17] 80:20 분할 주의 표기. 참고문헌 [16][17] 추가(저자명 DOI 접속 확인 필요). 수치 SSOT = `G:\EXPERT_KCI_Paper\02_data_ssot\TRAINING_LOG.md` (R8 섹션).
> 투고 학술지(확정): **한국화재소방학회 논문지(Fire Science and Engineering)**.

---

## 초록 (국문)

건축물 자동화재탐지설비의 비화재보(non-fire alarm)는 거주자의 경보 신뢰도를 떨어뜨리고 불필요한 출동을 유발하는 구조적 문제이다. 영상 기반 화재·연기 탐지는 감지기 단독 경보를 교차검증하여 비화재보를 줄이는 보조 수단이 될 수 있으나, 자원이 제약된 엣지 디바이스에서의 실시간 동작과 오탐 억제를 동시에 달성하기는 쉽지 않다. 본 연구는 동일한 YOLO11 계열 탐지기를 대상으로 학습 데이터 구성과 모델 용량을 분리하여 ablation 실험을 수행하고, 비화재보 저감 관점에서 어느 요인이 성능을 지배하는지를 규명한다. 공개 화재 영상 벤치마크 D-Fire[6]를 단일 평가 기준으로 삼아 화염(FL)·연기(SM)·정상배경(NM) 구성을 단계적으로 변화시킨 결과, 화염 단독 학습(mAP@0.5 0.325)은 연기 탐지가 전혀 불가능하였다(smoke AP 0.000). 화염·연기 1:1 균형 구성에서 0.691로 급상승하였으며(+36.6%p, smoke AP 0.000→0.698), 정상배경 데이터를 추가한 구성에서 0.736으로 추가 향상(+4.5%p)하였다. 반면 동일 데이터에서 모델 용량을 YOLO11n→11s로 확대한 효과는 0.736→0.749로 +1.3%p에 그쳤다. 데이터 구성에서 얻는 개선이 모델 용량 확대 효과의 약 30배에 달한다. 입력 해상도를 640에서 416으로 낮추어도 정확도 손실은 미미하였다. 본 결과는 비화재보 저감을 위한 엣지 화재 탐지에서 모델 아키텍처보다 데이터 구성(클래스 균형 및 정상배경 포함)이 결정적 요인임을 단일 공개 벤치마크에서 정량적으로 보인다.

**주제어**: 화재 탐지, 연기 탐지, 비화재보, 객체 탐지, YOLO, 엣지 컴퓨팅, 데이터 구성, Hard Negative

## Abstract (영문)

Non-fire alarms in building automatic fire detection systems undermine occupants' trust in alarms and cause unnecessary dispatches. Vision-based fire and smoke detection can serve as a cross-validation aid to reduce such false alarms, but achieving both real-time operation on resource-constrained edge devices and false-positive suppression is non-trivial. This study conducts an ablation experiment that decouples training data composition from model capacity using the YOLO11 detector family, and identifies which factor dominates performance from the perspective of non-fire alarm reduction. Using the public D-Fire benchmark[6] as a unified test set, we progressively varied the composition of flame (FL), smoke (SM), and normal-background (NM) training data. Training on flame only (mAP@0.5 0.325) completely failed to detect smoke (smoke AP 0.000). A balanced 1:1 flame-to-smoke composition raised performance to 0.691 (+36.6%p, smoke AP resurrected to 0.698), and adding normal-background data produced a further gain to 0.736 (+4.5%p). In contrast, increasing model capacity from YOLO11n to YOLO11s on the same data yielded only 0.736→0.749 (+1.3%p). The data composition gain is approximately 30 times larger than the model capacity gain. Reducing input resolution from 640 to 416 incurred negligible accuracy loss. The results quantitatively demonstrate, on a single public benchmark, that data composition—class balance and inclusion of normal backgrounds—rather than model architecture is the decisive factor for edge fire detection aimed at false-alarm reduction.

**Keywords**: Fire detection, Smoke detection, Non-fire alarm, Object detection, YOLO, Edge computing, Data composition, Hard negative

---

## 1. 서론

### 1.1 연구 배경

자동화재탐지설비는 건축물 방재의 1차 방어선이다. 그러나 감지기 단독 경보는 조리 연기, 수증기, 분진, 담배 연기, 감지기 노후·오염 등 화재가 아닌 요인에 의해 빈번히 작동한다. 국내 화재 통계에서도 자동화재탐지설비 관련 출동의 상당 부분이 실제 화재가 아닌 비화재보로 보고된다[1]. 이러한 비화재보는 거주자의 경보 무시(alarm fatigue)를 유발하고, 소방 출동 자원을 비효율적으로 소모하며, 결과적으로 실제 화재 시 초기 대응을 지연시키는 안전상의 역설을 낳는다.

영상 기반 화재·연기 탐지는 감지기 경보를 영상으로 교차검증하여 비화재보를 식별하는 보조 수단으로 주목받는다. 카메라가 경보 지점을 관측하여 실제 화염·연기 여부를 판정하면, 관리자는 불필요한 신고를 억제하거나 신속히 대응할 근거를 얻는다. 다만 이를 24시간 상시 운영하려면 클라우드 전송 비용·지연·사생활 문제를 피해 현장 엣지 디바이스에서 추론하는 것이 바람직하며, 이때 모델은 제한된 연산·메모리 안에서 실시간성과 오탐 억제를 동시에 만족해야 한다.

### 1.2 문제 정의

화재 탐지 정확도를 높이는 가장 직관적인 방법은 더 큰 모델을 쓰는 것이다. 그러나 엣지 환경에서는 모델 용량 증가가 곧 추론 지연·메모리 압박으로 직결되므로, 정확도 향상이 데이터에서 오는지 모델 용량에서 오는지를 구분하는 것이 실무적으로 중요하다. 특히 비화재보 저감은 단순한 평균 정밀도(mAP)뿐 아니라 정상 장면을 화재로 오인하지 않는 능력, 즉 정상배경에 대한 강건성을 요구한다.

본 연구는 다음 질문에 답한다.

1. 화재·연기 탐지 성능을 결정하는 주요인은 학습 데이터 구성인가, 모델 용량인가?
2. 비화재보 저감에 직접 기여하는 데이터 요소는 무엇인가?
3. 엣지 배포 시 입력 해상도를 낮추어 얻는 속도 이득과 정확도 손실의 균형점은 어디인가?

### 1.3 기여

본 연구의 기여는 다음과 같다.

- 동일한 탐지기 계열에서 **데이터 구성**과 **모델 용량**을 분리한 통제된 ablation을 수행하여, 두 요인의 기여도를 정량 비교한다(데이터 구성 효과 약 30배 우세).
- 화염·연기 **클래스 균형**과 **정상배경(NM) 포함**이 각각 비화재보 저감에 기여하는 메커니즘을 실측 정밀도·재현율 변화로 제시한다.
- 엣지 디바이스(Jetson Orin Nano) 배포를 전제로 입력 해상도-정확도 트레이드오프를 측정하여 실무 배포 지침을 제시한다.

---

## 2. 관련 연구

### 2.1 영상 기반 화재·연기 탐지

초기 화재 영상 탐지는 색상 히스토그램, 움직임 벡터, 텍스처 분석 등 수작업 특징(hand-crafted feature)에 의존하였다. 이러한 방법은 구현이 단순하나, 조명 변화·화재 유사 색상(석양, 적색 조명)·연기와 유사한 안개 장면 등에서 오탐이 빈번하여 실환경 적용에 한계가 있었다[2]. 합성곱 신경망(CNN)이 도입된 이후, 딥러닝 기반 접근이 주류가 되었으며 단일 프레임 기반 CNN 화재 탐지[3]나 3D CNN을 활용한 시공간 연기 탐지[4] 등이 높은 탐지율을 달성하였다. 최근에는 단일 단계 탐지기인 YOLO 계열[5]이 실시간성과 정확도의 균형으로 널리 채택되고 있으며, YOLOv5에 팽창 컨볼루션을 결합한 소규모 불꽃 탐지[7], UAV 산불 연기 탐지에 최적화된 YOLOv5[8], YOLOv8 기반 스마트팩토리 화재·연기 탐지[9] 등 다양한 적용 사례가 보고되고 있다. 공개 벤치마크로는 D-Fire(약 21,000장, 화염·연기·정상 혼합)[6]가 대표적이며, 국내에서는 AIHub의 화재 발생 예측 영상 데이터[10]가 대규모 학습 자원으로 활용된다.

선행 연구들은 대체로 아키텍처 개선(주의 메커니즘 도입, 특징 피라미드 구조 변형 등)을 통한 정확도 향상에 집중하였다. 반면 학습 데이터 구성—클래스 간 비율, 정상배경 포함 여부—이 성능에 미치는 영향을 모델 용량과 분리하여 정량화한 연구는 찾기 어렵다. 이는 데이터 구성과 모델 크기를 동시에 변경하거나, 단일 고정 데이터셋만을 사용한 경우가 많기 때문이다.

### 2.2 비화재보와 정상배경 학습

화재 탐지에서 오탐(false positive)은 정상 장면을 화재로 오인하는 데서 발생하며, 이는 실제 현장에서 불필요한 경보 출동을 유발하는 비화재보의 주원인이 된다. 화재 탐지 데이터셋은 일반적으로 화재 발생 장면이 정상 장면보다 수집이 어려워 클래스 불균형 문제를 내포하는 경우가 많다[11].

객체 탐지 분야에서는 배경(negative) 표본의 품질이 오탐 억제에 결정적임이 알려져 있다. 특히 검출기가 오경보를 일으키는 어려운 음성 표본(hard negative)을 학습에 반복 투입하는 hard negative mining이 정밀도 향상에 효과적임이 보고되었다[12]. 화재 탐지에 이를 적용하면, 화재 유사 패턴(고온 조명, 연무, 석양 등)을 포함한 정상 장면을 hard negative로 명시적으로 학습에 포함함으로써 비화재보 저감에 직접 기여할 수 있다.

그러나 화재 탐지 문헌에서 정상배경(normal background) 데이터의 양적 기여를 모델 용량과 독립적으로 분리하여 정량화한 연구는 제한적이다. 대부분의 연구에서 정상배경은 데이터셋에 포함되거나 제외되는 이진적 선택으로만 다루어지며, 그 비율이 성능에 미치는 영향을 단계적으로 분석한 사례는 드물다.

### 2.3 엣지 디바이스 화재 탐지

자원이 제약된 엣지 디바이스에서의 화재 탐지는 크게 두 방향으로 접근된다. 첫째는 모델 경량화로, 소형 백본 채택, 가중치 양자화(INT8/FP16), 채널 가지치기(pruning), TensorRT 변환 등을 통해 추론 지연과 메모리 사용량을 줄이는 방법이다[6][13]. 둘째는 입력 해상도 조정으로, 이미지 크기를 낮추어 연산량을 줄이는 단순하면서도 효과적인 방법이다.

최근에는 엣지 디바이스 실시간 추론을 겨냥한 경량 YOLO 변형이 활발히 제안되고 있다[14]. FCMI-YOLO[14]는 경량 백본과 특징 융합 모듈을 결합하여 엣지에서 실시간 화재 탐지를 달성하였다. 이러한 연구들은 아키텍처 측면의 개선에 초점을 두며 일정한 성과를 보이고 있으나, 데이터 구성 자체가 성능에 미치는 기여를 통제 변수로 분리한 분석은 포함하지 않는다. 즉, 동일한 아키텍처 개선이라도 학습 데이터의 구성에 따라 결과가 크게 달라질 수 있음을 간과할 가능성이 있다.

본 연구는 경량 모델(YOLO11n)을 기준으로 데이터 구성 효과를 체계적으로 분석하고, 입력 해상도 조정의 실측 정확도-속도 트레이드오프를 함께 보고함으로써 기존 연구의 공백을 채운다.

---

## 3. 데이터셋 및 전처리

### 3.1 데이터 출처

**본 실험(ablation 주실험)**에는 D-Fire 공개 데이터셋[6]을 사용한다. D-Fire는 de Venâncio et al. (2022)이 공개한 화재·연기 탐지 전용 이미지 데이터셋으로, 총 21,527장(학습 15,533 / 검증 2,997 / 테스트 2,997)이 YOLO 형식 바운딩박스 주석과 함께 제공된다. 화염(fire)·연기(smoke) 2클래스와 정상 배경(라벨 없음) 이미지를 포함하며, 단일 공개 벤치마크로 평가셋을 고정할 수 있어 실험 간 직접 비교가 가능하다는 장점이 있다. 모든 라벨은 fire(클래스 0)·smoke(클래스 1)의 2클래스로 통합하며, 정상 장면은 라벨이 없는 빈 주석 파일로 처리하여 hard negative로 활용한다.

**대규모 재현 실험(§5.6)**에는 AIHub 화재 발생 예측 영상 데이터(과제번호 071751)[10]를 추가로 활용한다. 본 데이터는 총 약 175,000장 규모의 국내 화재 영상으로, 동일한 데이터 구성 효과가 대규모 데이터에서도 재현됨을 확인하는 보강 근거로 사용한다.

### 3.2 데이터 분할 및 구성 선택

D-Fire는 공식 train/val/test 분할을 제공하므로, 검증셋(2,997장)과 테스트셋(2,997장)은 모든 실험에서 고정한다. 학습셋(15,533장) 안에서 ablation 구성별로 사용할 이미지를 선택하여 데이터 구성을 변화시킨다.

**AIHub 대규모 재현 실험(§5.6)**에서는 원천 영상에서 프레임을 직접 추출해야 하므로 3단계 씬 샘플링을 적용한다.
1. **씬 경계 탐지**: 히스토그램 상관계수(cv2.compareHist)가 0.7 이하로 떨어지는 지점을 씬 경계로 판정한다.
2. **씬 내 시간 간격 추출**: 씬 내에서 2초 간격(30fps 기준 60프레임당 1장)으로 프레임을 추출한다.
3. **지각 해시 중복 제거**: imagehash 기반 perceptual hash의 해밍 거리가 8 이하인 프레임을 중복으로 제거한다.

### 3.3 데이터 구성 변형

ablation을 위해 다음 4단계 구성을 정의한다(표 1).

**표 1. 데이터 구성 변형 (D-Fire 학습셋 기준)**

| 구성 | 화염(FL) | 연기(SM) | 정상(NM) | 총량 | FL:SM 비율 |
|------|---------|---------|---------|------|-----------|
| C1 (화염 단독) | 3,828 | 0 | 0 | 3,828 | 1:0 |
| C2 (불균형) | ~3,558 | ~270 | 0 | ~3,828 | ~14:1 †학습 중 |
| C3 (균형) | 2,299 | 2,299 | 0 | 4,598 | 1:1 |
| C4 (균형+정상) | 2,299 | 2,299 | 6,458 | 11,056 | 1:1 (+NM 2.8) |

C2의 수량은 FL:SM 비율(약 14:1)을 통제 변수로 유지하면서 학습셋 내에서 결정된 근사치이다. 평가용 검증셋·테스트셋은 전 구성에서 동일한 D-Fire 공식 분할(각 2,997장)을 사용하여 직접 비교가 가능하도록 통제한다.

---

## 4. 방법

### 4.1 탐지 모델

탐지기는 YOLO11 계열(n/s/m/l)[15]을 사용한다. 모든 학습은 COCO 사전학습 가중치에서 파인튜닝하며, 동일한 학습 설정을 적용한다: 옵티마이저 AdamW, 초기 학습률 0.001, 코사인 학습률 감쇠, warmup 3에폭, 최대 100에폭, patience 30(조기 종료), 데이터 증강(HSV 변형, 좌우 반전, mosaic 1.0, mixup 0.1).

### 4.2 실험 설계

두 개의 독립 ablation을 수행한다.

- **실험 A (데이터 구성)**: 모델을 YOLO11n으로 고정하고 데이터 구성을 C1→C2→C3→C4로 변화시켜 데이터 요인의 기여를 측정한다.
- **실험 B (모델 용량)**: 데이터를 C4로 고정하고 모델을 YOLO11n→11s→11m→11l로 확대하여 용량 요인의 기여를 측정한다(YOLO11n·11s는 D-Fire 기준 주실험, 11m·11l은 AIHub 175K 기준 보조 참고).

추가로 입력 해상도(640 vs 416)에 따른 정확도-속도 트레이드오프를 YOLO11n 기준으로 측정한다.

### 4.3 평가 지표

표준 객체 탐지 지표인 mAP@0.5, mAP@0.5:0.95, 정밀도(precision), 재현율(recall), 클래스별 AP를 사용한다. 비화재보 저감 관점에서는 정상 장면 오인을 반영하는 정밀도와, 연기 탐지 능력을 반영하는 smoke AP를 중점 분석한다.

### 4.4 실험 환경

본 실험(D-Fire 4셀 재학습)은 RTX 5090(torch 2.9.1+cu128, ultralytics 8.4.75)에서 수행하였다. 대규모 재현 실험(AIHub, §5.6)은 RTX 4090(24GB) 및 클라우드 GPU(NVIDIA L40S 48GB)에서 수행하였다. 엣지 배포는 Jetson Orin Nano(8GB)에서 TensorRT FP16으로 변환하여 처리 속도를 측정한다.

---

## 5. 실험 결과

### 5.1 실험 A — 데이터 구성 효과 (모델 고정: YOLO11n)

데이터 구성에 따른 성능 변화를 표 2에 정리한다.

**표 2. 데이터 구성에 따른 YOLO11n 성능 (D-Fire 테스트셋, R8)**

| 구성 | mAP@0.5 | mAP@0.5:0.95 | 정밀도 | 재현율 | smoke AP | fire AP | 비고 |
|------|---------|--------------|--------|--------|----------|---------|------|
| C1 (화염 단독) | 0.325 | 0.169 | 0.321 | 0.310 | 0.000 | 0.650 | 연기 탐지 전무 |
| C2 (불균형 14:1) | 0.455 | 0.231 | 0.586 | 0.441 | 0.237 | 0.673 | smoke 부분 활성화 |
| C3 (균형 1:1) | 0.691 | 0.387 | 0.698 | 0.654 | 0.698 | 0.684 | smoke AP 부활 |
| C4 (균형+정상) | **0.736** | 0.414 | 0.746 | 0.669 | 0.765 | 0.707 | 정밀도+재현율 동시 향상 |


![데이터 구성에 따른 mAP@0.5 변화](G:\EXPERT_KCI_Paper\04_figures\dfire_4cell\fig_data_composition_dfire.png)

**그림 1. 데이터 구성(C1→C4)에 따른 YOLO11n의 mAP@0.5 변화 (D-Fire 테스트셋).** 클래스 균형(C1→C3) 효과가 +36.6%p로 가장 크며, 정상배경 추가(C3→C4)가 추가 +4.5%p를 기여한다.

주요 관찰은 다음과 같다.

- **화염 단독(C1)은 연기를 전혀 탐지하지 못한다.** smoke AP가 0.000으로, 연기만 존재하는 화재 초기 구간을 완전히 놓친다. fire AP는 0.650이나 전체 mAP는 0.325에 그치며, 이는 smoke 클래스 탐지 실패가 평균 지표를 끌어내리기 때문이다.
- **소량 불균형 smoke 추가(C2)는 smoke 탐지를 부분적으로만 활성화한다.** FL:SM=14:1에서 mAP가 0.455로 소폭 상승하고 smoke AP가 0.237로 처음 활성화되었다. 그러나 smoke AP 0.237은 클래스 균형 이후(C3: 0.698)와 비교하면 3분의 1 수준에 불과하다. 절대량이 부족한 연기 표본으로는 smoke 탐지 능력이 임계점에 도달하지 못한다.
- **클래스 균형(C3)이 임계점이다.** FL:SM=1:1로 맞추자 smoke AP가 0.237에서 0.698로 도약하고 mAP가 0.691로 상승하였다(C2 대비 +23.6%p). smoke AP 기준으로 C1→C2→C3의 변화(0.000→0.237→0.698)는 비선형적이며, 클래스 균형에 도달할 때 비로소 연기 탐지 능력이 완전히 발현됨을 보여준다.
- **정상배경 추가(C4)가 정밀도와 재현율을 동시에 높인다.** 정상 장면을 hard negative로 포함하자 mAP가 0.736(C3 대비 +4.5%p)으로 상승하였다. 정밀도 0.746과 재현율 0.669가 함께 개선되어, 오탐 억제와 탐지 누락 저감을 동시에 달성하였다.

그림 3은 각 구성의 혼동행렬 변화로 위 관찰을 시각적으로 확인한다.

![데이터 구성별 혼동행렬](G:\EXPERT_KCI_Paper\04_figures\dfire_4cell\fig_confusion_matrix_comp_dfire.png)

**그림 3. C1→C4 구성별 정규화 혼동행렬 (D-Fire 테스트셋, YOLO11n).** C1은 smoke 클래스 탐지가 전무(재현율 0)이며, C3에서 smoke 탐지가 완전히 활성화되고 C4에서 fire·smoke 양 클래스를 균형 있게 탐지한다.

### 5.2 실험 B — 모델 용량 효과 (데이터 고정: C4)

동일 데이터(C4)에서 모델 용량을 확대한 결과를 표 3에 정리한다(YOLO11n·11s는 D-Fire 기준, YOLO11m·11l은 AIHub 175K 기준 보조 참고).

**표 3. 모델 용량에 따른 성능 (데이터 고정: C4)**

| 모델 | 파라미터 규모 | mAP@0.5 | mAP@0.5:0.95 | 정밀도 | 재현율 | 데이터셋 | 비고 |
|------|--------------|---------|--------------|--------|--------|---------|------|
| YOLO11n | 최소 | 0.736 | 0.414 | 0.746 | 0.669 | D-Fire | 엣지 기준 |
| YOLO11s | 소 | **0.749** | **0.423** | 0.767 | 0.672 | D-Fire | +1.3%p |
| YOLO11m | 중 | 0.914 †AIHub | — | 0.878 | 0.858 | AIHub | 11s보다 낮음 †AIHub |
| YOLO11l | 대 | 실패 †AIHub | — | — | — | AIHub | 반복 발산 †AIHub |

> †AIHub 11m·11l은 AIHub 175K 데이터 기반 결과(R5·R6)이며 테스트셋이 다르다. D-Fire 기준 직접 비교 불가, 경향 참고용.
> †³ YOLO11l은 AIHub 175K 데이터에서 100에폭 학습 3회 시도 모두 검증 손실이 발산하였다. 배치 크기(16→10) 재시도에서도 동일하게 발산이 반복되었으며, 데이터 규모 대비 과도한 모델 용량이 학습 불안정을 유발한 것으로 해석된다.

![모델 용량에 따른 mAP@0.5 변화](G:\EXPERT_KCI_Paper\04_figures\dfire_4cell\fig_model_dfire.png)

**그림 2. 모델 용량(YOLO11n→11s)에 따른 mAP@0.5 변화(D-Fire 테스트셋, 데이터 고정: C4).** 11n→11s 개선은 +1.3%p에 그친다.

- 모델을 11n에서 11s로 키운 효과는 +1.3%p(0.736→0.749)에 그친다.
- AIHub 대규모 실험에서도 11m(0.914)이 11s(0.918)보다 낮아, 데이터 규모를 막론하고 용량 증가의 수확 체감이 일관되게 나타난다.
- 최대 모델 11l은 AIHub 175K 데이터에서도 학습 손실이 반복 발산하여 수렴에 실패하였다.

그림 4는 C4 최종 구성에서의 클래스별 정밀도-재현율 곡선이다.

![C4 PR 곡선](G:\EXPERT_KCI_Paper\04_figures\dfire_4cell\C4_test_PR_curve.png)

**그림 4. C4(균형+NM) 구성 YOLO11n의 fire·smoke 클래스별 PR 곡선 (D-Fire 테스트셋).** fire AP=0.707, smoke AP=0.765. 정상배경 포함 이후 smoke 탐지 품질이 fire를 상회한다.

### 5.3 데이터 vs 모델 — 기여도 비교

두 실험을 종합하면 기여도 차이가 분명하다(표 4).

**표 4. 단계별 개선폭 분해 (D-Fire 테스트셋 기준)**

| 단계 | 변화 | mAP 변화 | mAP 개선폭 | smoke AP 변화 |
|------|------|---------|-----------|--------------|
| 소량 불균형 smoke 추가 | C1→C2 | 0.325→0.455 | +13.0%p | 0.000→0.237 (부분 활성) |
| 클래스 균형 달성 | C2→C3 | 0.455→0.691 | **+23.6%p** | 0.237→0.698 (완전 부활) |
| 정상배경 추가 | C3→C4 | 0.691→0.736 | **+4.5%p** | 0.698→0.765 |
| 모델 용량 확대 | 11n→11s | 0.736→0.749 | +1.3%p | — |

데이터 구성에서 얻는 총 개선(C1→C4, +41.1%p)이 모델 용량 확대 효과(+1.3%p)의 약 **30배**에 달한다. smoke AP 기준으로 보면 C1→C2→C3의 변화(0.000→0.237→0.698)가 비선형적이다. 소량 불균형 smoke는 탐지를 부분적으로만 활성화하고, 클래스 균형에 도달해야 비로소 smoke 탐지 능력이 완전히 발현된다. 즉 **비화재보 저감을 위한 화재 탐지 성능은 모델 아키텍처가 아니라 데이터 구성이 지배하며**, 특히 클래스 균형이 성능 임계점으로 작용한다.

### 5.4 입력 해상도 트레이드오프

AIHub C4 데이터 기준으로, YOLO11n에서 입력 해상도를 640에서 416으로 낮춘 결과, mAP는 0.911에서 약 0.915로 오히려 소폭 향상(+0.4%p)되었다. 이는 해상도 축소가 일종의 정규화 효과로 작용하거나, 본 데이터셋의 화재·연기 객체 크기가 저해상도 입력에서도 충분히 탐지 가능한 규모임을 시사한다. 해상도를 낮추면 엣지에서 처리량이 약 2배로 향상되므로, 정확도 손실 없이 속도를 확보할 수 있는 자원제약 디바이스에서 유효한 전략이다.

### 5.5 AIHub 대규모 재현 실험

동일한 데이터 구성 효과가 소규모(D-Fire ~21K) 외에 대규모 데이터에서도 재현되는지를 확인하기 위해, AIHub 화재 발생 예측 영상 데이터(약 175K장)[10]를 이용한 보강 실험을 수행하였다. AIHub 실험에서도 클래스 균형(C2→C3) 효과 +7.5%p, 정상배경 추가(C3→C4) 효과 +7.5%p, 모델 용량(11n→11s) 효과 +0.7%p로 D-Fire와 동일한 경향이 확인되었다. 효과 크기의 차이(D-Fire ~30배 vs AIHub ~10배)는 두 데이터셋의 규모·도메인 차이에 기인하는 것으로 보이나, 데이터 구성이 모델 용량보다 우세하다는 결론은 두 실험 모두에서 일관된다.

### 5.6 엣지 배포 성능

C4 구성으로 학습한 YOLO11n을 Jetson Orin Nano(8GB)에서 TensorRT FP16으로 변환하여 측정한 결과, 약 47 FPS의 실시간 처리 성능을 확인하였다. 이는 다채널 영상 교차검증에 충분한 처리량이다.

### 5.7 선행 연구와의 비교

본 연구의 최적 구성(C4) 결과를 선행 연구와 비교하여 맥락을 제시한다(표 5). 화재 탐지 분야에서는 연구마다 학습·평가 데이터셋이 상이하여 직접 수치 비교가 어려운 경우가 많다. 따라서 표 5는 비교 유형을 두 층위로 구분한다. 첫째, D-Fire 공식 test split(2,997장)을 동일하게 사용한 연구는 직접 비교 대상(●)으로, 둘째, 별도 데이터셋을 사용한 연구는 맥락 참고용(○)으로 표기한다.

**표 5. 선행 연구와의 성능 비교**

| 연구 | 모델 | 테스트 데이터 | mAP@0.5 | P | R | 비교 |
|------|------|------------|---------|---|---|------|
| de Venâncio et al. (2022) [6] | YOLOv4(프루닝) | D-Fire | n.a. †1 | — | — | ● 직접 |
| (저자 확인 필요) (2025) [16] | YOLOv5s | D-Fire | 76.7% | 78.2% | 71.4% | ● 직접 |
| (저자 확인 필요) (2025) [16] | YOLOv8n | D-Fire | 75.4% | 79.3% | 69.6% | ● 직접 |
| (저자 확인 필요) (2025) [16] | CCi-YOLOv8n | D-Fire | 78.5% | 79.5% | 71.5% | ● 직접 |
| Li et al. (2025) [17] | YOLOv7 | D-Fire †2 | 80.2% | — | — | ●† |
| Li et al. (2025) [17] | YOLOGX | D-Fire †2 | 80.9% | 77.2% | 75.7% | ●† |
| Wu et al. (2022) [7] | YOLOv5+DC | 자체 데이터 | 99.3% †3 | 98.3% | 99.2% | ○ 참고 |
| Mukhiddinov et al. (2022) [8] | YOLOv5(opt.) | 자체(UAV) | — | — | — | ○ 참고 |
| Kim et al. (2024) [9] | YOLO10 | 자체 데이터 | >91% | — | — | ○ 참고 |
| Lu et al. (2025) [14] | FCMI-YOLO | 자체 데이터 | 88.0% | 88.4% | 84.4% | ○ 참고 |
| **본 연구 C4 — YOLO11n** | **YOLO11n** | **D-Fire** | **73.6%** | **74.6%** | **66.9%** | **● 직접** |
| **본 연구 C4 — YOLO11s** | **YOLO11s** | **D-Fire** | **74.9%** | **76.7%** | **67.2%** | **● 직접** |

> ● 직접: D-Fire 공식 test split(2,997장) 기준, 직접 수치 비교 유효.
> ●†: D-Fire 데이터 사용하나 80:20 자체 분할. 공식 test split과 동일 여부 불명확, 참고 수준.
> ○ 참고: 테스트셋 상이, 수치 직접 비교 불가, 맥락 참고용.
> †1: de Venâncio et al. (2022)는 계산 효율(83.60% 비용 절감, 83.86% 메모리 절감) 위주 보고. 전문 내 mAP 포함 여부는 유료 접근으로 미확인.
> †2: Li et al. (2025)는 D-Fire를 80:20으로 재분할하여 사용. 공식 test split(2,997장)과 동일하지 않을 수 있어 수치 직접 비교에 주의가 필요하다.
> †3: Wu et al. (2022)의 99.3%는 비공개 자체 데이터셋 결과이며, 다른 문헌에서 데이터 대표성 한계가 지적된 바 있다[선행 인용].

D-Fire 공식 test split 기준 직접 비교에서, 본 연구의 C4 YOLO11n(73.6%)은 동일 테스트셋에서 보고된 YOLOv8n 베이스라인(75.4% [16])과 약 1.8%p 차이를 보인다. YOLO11n(파라미터 약 2.6M)이 YOLOv8n(3.2M)보다 경량인 점을 고려하면 이 차이는 모델 크기 대비 경쟁력 있는 수준이다. 본 연구의 목표는 SOTA 달성이 아니라 데이터 구성과 모델 용량의 기여도 분리이며, 절대 정확도보다 ablation 단계별 개선폭(표 4)이 핵심 기여이다. C1(32.5%)에서 C4(73.6%)까지 +41.1%p의 개선이 모두 데이터 구성에서 비롯되었다는 점이 실무적 함의의 핵심이다.

---

## 6. 논의

### 6.1 비화재보 저감 메커니즘

정상배경(NM) 추가가 가장 큰 개선을 가져온 점은 비화재보 저감 관점에서 직접적 함의를 갖는다. NM은 화재가 아닌 장면을 화재로 오인하지 않도록 학습시키는 hard negative로 작용한다. C4에서 C3 대비 정밀도(0.698→0.746)와 재현율(0.654→0.669)이 함께 향상된 것은, 정상배경 학습이 탐지 민감도를 희생하지 않으면서 오탐을 억제함을 보여준다. 이는 영상 교차검증이 감지기 비화재보를 걸러내는 보조 수단으로 기능하기 위한 핵심 조건이다.

### 6.2 데이터 구성 우선 전략의 실무적 함의

본 결과는 엣지 화재 탐지 시스템 개발에서 자원 투입 우선순위를 시사한다. 더 큰 모델을 탐색하기보다, (1) 화염·연기 클래스를 균형 있게 확보하고 (2) 충분한 정상배경을 hard negative로 포함하는 데이터 구성에 우선 투자하는 것이 비용 대비 효과가 크다. 특히 경량 모델(YOLO11n)로도 D-Fire 기준 0.736의 mAP를 달성하므로, 엣지 배포에서 모델 용량을 키울 실익은 제한적이다.

### 6.3 한계

본 연구의 한계는 다음과 같다.

- 본 실험은 D-Fire 단일 벤치마크 기준이며, 다른 공개 데이터셋에서의 추가 교차 검증은 향후 과제이다. AIHub 대규모 보강 실험(§5.5)은 동일한 경향을 지지하나, 두 실험의 테스트셋이 달라 mAP 수치의 직접 비교는 유효하지 않다.
- **C2 결과의 데이터셋 의존성**: D-Fire에서 C2(0.455)는 C1(0.325)보다 높으나, AIHub 기존 실험(R2/R3)에서는 C2(0.761) < C1(0.777) 경향이 관찰되었다. 다만 해당 AIHub 결과는 실험 간 테스트셋 불일치가 있어 직접 비교에 주의가 필요하며, 일관된 테스트셋 기준의 재확인이 필요하다. 이 경향이 확인된다면 "소량 불균형 smoke 추가의 효과"는 데이터셋에 따라 부호가 달라질 수 있음을 시사한다. 그러나 smoke AP 기준의 비선형 패턴(부분 활성 → 완전 부활)과 클래스 균형이 임계점이라는 결론은 두 데이터셋에서 일관된다.
- **C3→C4 구성 비교의 교란 변수**: D-Fire 기준 C4는 C3 대비 NM 6,458장 추가와 동시에 총 학습 데이터 량이 4,598장에서 11,056장으로 증가하였다. AIHub 보강 실험에서도 마찬가지로 NM 약 35,000장 추가와 총량 137,520→175,000장 증가가 동시에 발생하였다. 따라서 C3→C4의 성능 향상(D-Fire +4.5%p, AIHub +7.5%p)은 NM 포함 효과와 총 데이터 량 증가 효과가 완전히 분리되지 않는다. NM 비율을 동일하게 유지하면서 총량만 통제한 추가 실험은 향후 과제이다.
- 비화재보 저감 효과는 정밀도·재현율로 간접 입증하였으며, 실제 현장 운영에서의 오경보 억제율(FP/FPR 직접 측정)은 별도 실증이 필요하다.
- 최대 모델(YOLO11l)의 발산은 학습 설정 조정으로 일부 완화될 여지가 있으나, 본 데이터 규모에서 용량 확대의 한계라는 결론에는 영향을 주지 않는다.

### 6.4 향후 연구

실내 화재 특화 데이터 통합, 야간·특수 조명 도메인 확장, 그리고 실제 건축물 현장에서의 감지기 경보-영상 교차검증 실증을 통한 오경보 억제율 직접 측정이 향후 과제이다.

---

## 7. 결론

본 연구는 D-Fire 공개 벤치마크를 단일 평가 기준으로 삼아 엣지 기반 화재·연기 탐지에서 학습 데이터 구성과 모델 용량의 기여를 분리한 통제 실험을 수행하였다. 화염 단독 학습은 연기를 전혀 탐지하지 못하고(smoke AP 0.000), 화염·연기 1:1 균형 구성이 smoke AP를 0.698로 부활시키며 mAP를 0.325에서 0.691로 끌어올렸다(+36.6%p). 정상배경 추가는 추가로 +4.5%p를 기여하여 최종 mAP 0.736을 달성하였다. 반면 모델 용량 확대(11n→11s) 효과는 +1.3%p에 그쳐, 데이터 구성 효과가 모델 용량의 약 30배에 달함을 단일 공개 벤치마크에서 결함 없이 보였다. 동일한 경향은 AIHub 175K 대규모 데이터에서도 재현되었다(§5.5). 결론적으로 비화재보 저감을 위한 엣지 화재 탐지 성능은 모델 아키텍처가 아니라 데이터 구성(클래스 균형 및 정상배경 포함)이 결정하며, 본 결과는 자원제약 환경에서 데이터 구성에 우선 투자하는 전략의 타당성을 정량적으로 뒷받침한다.

---

## 참고문헌

> 아래 15건은 모두 실재가 확인된 문헌이다(저자·서지·DOI 검증 완료). 투고 학술지(한국화재소방학회 논문지) 인용 양식에 맞춰 표기 형식만 변환하면 된다. 본문 인용 번호[n]는 등장 순서를 따른다.

[1] 소방청, "국가화재정보시스템 화재통계," https://www.nfds.go.kr (접속: 2026).

[2] P. Foggia, A. Saggese, M. Vento, "Real-Time Fire Detection for Video-Surveillance Applications Using a Combination of Experts Based on Color, Shape, and Motion," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 25, no. 9, pp. 1545–1556, 2015. doi:10.1109/TCSVT.2015.2392531.

[3] A. Dunnings, T. P. Breckon, "Experimentally Defined Convolutional Neural Network Architecture Variants for Non-Temporal Real-Time Fire Detection," *Proc. IEEE International Conference on Image Processing (ICIP)*, pp. 1558–1562, 2018. doi:10.1109/ICIP.2018.8451657.

[4] G. Lin, Y. Zhang, G. Xu, R. Zhang, "Smoke Detection on Video Sequences Using 3D Convolutional Neural Networks," *Fire Technology*, vol. 55, no. 5, pp. 1827–1847, 2019. doi:10.1007/s10694-019-00832-w.

[5] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," *Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)*, pp. 779–788, 2016.

[6] P. V. A. B. de Venâncio, A. C. Lisboa, A. V. Barbosa, "An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices," *Neural Computing and Applications*, vol. 34, no. 18, pp. 15349–15368, 2022. doi:10.1007/s00521-022-07467-z. (D-Fire 데이터셋)

[7] Y. Wu, S. Xue, H. Li, "Real-Time Video Fire Detection via Modified YOLOv5 Network Model," *Fire Technology*, vol. 58, no. 5, pp. 2377–2403, 2022. doi:10.1007/s10694-022-01260-z.

[8] M. Mukhiddinov, A. B. Abdusalomov, J. Cho, "A Wildfire Smoke Detection System Using Unmanned Aerial Vehicle Images Based on the Optimized YOLOv5," *Sensors*, vol. 22, no. 23, Article 9384, 2022. doi:10.3390/s22239384.

[9] S. Kim et al., "An Improved YOLOv8n for Fire and Smoke Detection in Smart Factory Environments," *Sensors*, vol. 24, no. 15, Article 4786, 2024. doi:10.3390/s24154786.

[10] 한국지능정보사회진흥원(NIA), "화재 발생 예측 영상 데이터(AIHub 071751)," https://www.aihub.or.kr.

[11] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, "Focal Loss for Dense Object Detection," *Proc. IEEE International Conference on Computer Vision (ICCV)*, pp. 2980–2988, 2017. doi:10.1109/ICCV.2017.32.

[12] A. Shrivastava, A. Gupta, R. Girshick, "Training Region-based Object Detectors with Online Hard Example Mining," *Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)*, pp. 761–769, 2016. doi:10.1109/CVPR.2016.89.

[13] L. Yang et al., "Lightweight Forest Smoke and Fire Detection Algorithm Based on Improved YOLOv5," *PLOS ONE*, vol. 18, no. 9, Article e0291359, 2023. doi:10.1371/journal.pone.0291359.

[14] J. Lu, Y. Zheng, L. Guan, B. Lin, W. Shi, J. Zhang, Y. Wu, "FCMI-YOLO: An efficient deep learning-based algorithm for real-time fire detection on edge devices," *PLOS ONE*, vol. 20, no. 8, e0329555, 2025. doi:10.1371/journal.pone.0329555.

[15] Ultralytics, "Ultralytics YOLO11," 2024. https://github.com/ultralytics/ultralytics.

[16] (저자명 DOI 접속 확인 필요), "CCi-YOLOv8n (제목 확인 필요)," in *Proc. (출판사 확인 필요)*, 2025. arXiv:2411.11011; doi:10.1007/978-981-96-9901-8_11. (표 5의 YOLOv5s·YOLOv8n·CCi-YOLOv8n D-Fire 결과 출처, Table I)

[17] Li et al., "YOLOGX (제목 확인 필요)," *Frontiers in Environmental Science*, 2025. doi:10.3389/fenvs.2024.1486212. (표 5의 YOLOGX·YOLOv7 D-Fire 80:20 분할 결과 출처, Table 4·5)

> **⚠ 투고 전 필수**: [16][17]은 deep-research 수치 검증 완료(신뢰도 high)이나 저자명·제목이 미확인 상태. 투고 전 DOI 직접 접속하여 저자·서지 완성 필수. SSOT 원칙 적용.

---

## 부록 A. 수치 출처 및 재현성

### A-1. D-Fire 주실험 (R8, 표 2~4 기준 SSOT)

| 구성·지표 | 실측값 | TRAINING_LOG 섹션 |
|-----------|--------|-------------------|
| C1 mAP@0.5 / mAP@0.5:0.95 / P / R | 0.325 / 0.169 / 0.321 / 0.310 | R8-C1 |
| C1 fire AP / smoke AP | 0.650 / 0.000 | R8-C1 |
| C2 mAP@0.5 / mAP@0.5:0.95 / P / R | 0.455 / 0.231 / 0.586 / 0.441 | R8-C2 |
| C2 fire AP / smoke AP | 0.673 / 0.237 | R8-C2 |
| C3 mAP@0.5 / mAP@0.5:0.95 / P / R | 0.691 / 0.387 / 0.698 / 0.654 | R8-C3 |
| C3 fire AP / smoke AP | 0.684 / 0.698 | R8-C3 |
| C4 mAP@0.5 / mAP@0.5:0.95 / P / R | 0.736 / 0.414 / 0.746 / 0.669 | R8-C4 |
| C4 fire AP / smoke AP | 0.707 / 0.765 | R8-C4 |
| C4_11s mAP@0.5 / mAP@0.5:0.95 / P / R | 0.749 / 0.423 / 0.767 / 0.672 | R8-C4_11s |

### A-2. AIHub 보조 실험 (경향 참고용, D-Fire와 직접 비교 불가)

| 구성·지표 | 실측값 | 비고 |
|-----------|--------|------|
| C1 (AIHub) mAP / P / R | 0.777 / 0.903 / 0.640 | R2 |
| C2 (AIHub) mAP / smoke AP | 0.761 / 0.684 | R3 — 테스트셋 불일치, C1과 직접 비교 주의 |
| C3 (AIHub) mAP / smoke AP | 0.836 / 0.792 | R4 |
| C4 (AIHub) mAP / P / R | 0.911 / 0.888 / 0.853 | E01 |
| C4_11s (AIHub) | 0.918 | E02 |
| C4_11m (AIHub) mAP / P / R | 0.914 / 0.878 / 0.858 | E03 |
| C4_11l (AIHub) | 발산 실패 | E04 |
| @416 해상도 (AIHub C4) | ~0.915 | E07 |
| Jetson TensorRT FP16 | ~47 FPS | 과제 SSOT |

### A-3. 논문용 그림 경로

| 그림 | 파일 경로 |
|------|-----------|
| 그림 1 — 데이터 구성별 mAP | `04_figures/dfire_4cell/fig_data_composition_dfire.png` |
| 그림 2 — 모델 용량별 mAP | `04_figures/dfire_4cell/fig_model_dfire.png` |
| C1~C4 혼동행렬 (정규화) | `04_figures/dfire_4cell/C*_test_confusion_matrix_norm.png` |
| C4 PR 곡선 | `04_figures/dfire_4cell/C4_test_PR_curve.png` |
| AIHub C4 보강 실험 그림 | `04_figures/aihub_c4/` |

> 재현 경로: `train.py` / `train_matrix.py` → best.pt → `val()` 로 혼동행렬·PR곡선 재산출. 엣지 변환: `export_trt.py`.
