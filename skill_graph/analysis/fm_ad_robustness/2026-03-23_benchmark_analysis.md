# Robustness Benchmark 기술 분석: MVTec AD 2 & RobustAD

> 작성일: 2026-03-23
> 목적: 연구 대상 벤치마크의 정밀 기술 분석 — 실험 설계의 근거 자료

---

## 1. MVTec AD 2 (arXiv:2503.21622, March 2025)

### 1.1 기본 정보
- **저자**: Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca Konig, Carsten Steger (MVTec Software GmbH)
- **총 이미지 수**: 8,004+ high-resolution images
- **카테고리**: 8개 산업 검사 시나리오
- **라이선스**: CC BY-NC-SA 4.0 (비상업적 용도만 허용)
- **데이터 다운로드**: mvtec.com/company/research/datasets/mvtec-ad-2
- **평가 서버**: benchmark.mvtec.com (현재 VAND 4.0 @ CVPR 2026 준비 중, 2026-04-01 재오픈 예정)
- **연관 챌린지**: CVPR 2025 VAND 3.0 Workshop Challenge Track 1: "Adapt & Detect: Robust Anomaly Detection in Real-World Applications"

### 1.2 8개 카테고리 상세

| 카테고리 | 핵심 도전 | 특징 |
|---------|---------|------|
| **Can** | 반사면(reflective surface), 기하학적 패턴 변형 | 높은 intra-class 변동, SOTA 방법 30% 미만 |
| **Fabric** | 다양한 텍스처, 배경 위 fabric 겹침 | 텍스처 일관성 모델링 필요, edge anomaly |
| **Fruit Jelly** | 투명/반투명 매질, 에어버블 다양성 | Backlighting, 재료 분포 평가 필요, implicit structural consistency |
| **Rice** | 극소 결함 (이미지 면적 < 0.1%), 플라스틱 오염물 | 고해상도 필수, subpixel 수준 검출 |
| **Sheet Metal** | Dark-field 조명, 거울 반사, 긴 종횡비 | Grayscale, 비균일 aspect ratio |
| **Vial** | 투명 유리 + 액체 굴절, 결함과 반사 구분 | Backlighting, missing-type anomaly 검출 어려움 |
| **Wallplugs** | 벌크 겹침 객체, 랜덤 오클루전, 경계 잘림 | 복잡한 배경, boundary anomaly, semantic understanding 필요 |
| **Walnuts** | 벌크 겹침 객체, 불규칙 형상 | 랜덤 오클루전, edge anomaly |

### 1.3 이미지 사양
- **해상도**: 1400x1900 ~ 2448x2048 (전 카테고리 high-resolution)
- **특이사항**:
  - 일부 카테고리 grayscale (Sheet Metal 등)
  - 비균일 aspect ratio (Sheet Metal은 elongated)
  - 20% 이상의 anomaly가 281 픽셀 미만, 최소 5 픽셀 수준의 anomaly 존재

### 1.4 데이터셋 구조 (3-tier test set)

```
Train:       정상 이미지, 일정 조명 조건
TESTpub:     공개 GT 포함, 혼합 조명 조건, 예비 검증용 (limited annotated samples)
TESTpriv:    비공개 GT, 학습 도메인 조명 조건 유지 → 표준 평가
TESTpriv,mix: 비공개 GT, seen + unseen 조명 조건 혼합 → 로버스트니스 평가
```

### 1.5 Multi-Lighting 평가 프로토콜

**핵심 설계**:
- 각 시나리오에 **최소 4가지 조명 조건** 포함: regular, underexposed, overexposed, additional light sources
- 실제 생산 환경의 장비 변동 시뮬레이션
- **TESTpriv**: 학습 시와 동일한 조명 → 표준 AD 성능
- **TESTpriv,mix**: seen + unseen 조명 혼합 → 실전 로버스트니스
- **Robustness 측정**: TESTpriv 대비 TESTpriv,mix의 성능 변화 (degradation)

**의미**: 단순히 "다른 데이터"가 아니라, 동일한 결함이 다른 조명에서 어떻게 보이는지를 체계적으로 평가. 이는 실제 공장에서 조명 장비가 교체/노화되는 상황을 반영.

### 1.6 AU-PRO 메트릭

**Per-Region Overlap (PRO)**: Bergmann et al. (IJCV 2021)에서 정의.

**원리**:
- 각 연결된 anomaly region에 대해 개별적으로 overlap을 계산
- 큰 anomaly나 작은 anomaly를 동등하게 취급 (size-independent)
- AUROC는 큰 defect에 지배됨 → PRO는 이를 해결

**AU-PRO 계산**:
- PRO curve: FPR(false positive rate)을 x축, 평균 per-region overlap을 y축으로 plot
- AU-PRO = PRO curve 아래의 면적을 [0, FPR_max]까지 적분 후 정규화

**AU-PRO(0.05) vs AU-PRO(0.3)**:
- `AU-PRO(0.3)`: MVTec AD 원본에서 사용. FPR 0~0.3 구간 적분. 상대적으로 관대.
- `AU-PRO(0.05)`: MVTec AD 2에서 채택. FPR 0~0.05 구간 적분. **6배 더 엄격**.
  - **의미**: 극히 낮은 false positive 영역에서의 성능만 평가
  - **이유**: 산업 환경에서 false positive는 불필요한 생산 중단을 초래 → 실전에서는 FPR << 5%여야 유의미
  - **효과**: 기존 방법의 높은 성능이 크게 하락 — AUROC가 높아도 AU-PRO(0.05)는 낮을 수 있음
  - **실례**: MVTec AD에서 AU-PRO(0.3) 97%+ 달성하던 방법이 MVTec AD 2에서 AU-PRO(0.05) 60% 미만

### 1.7 벤치마크 결과 — 전체 비교표

#### Table A: SegF1 (%) — TESTpriv / TESTpriv,mix

| Object | PatchCore | RD | RD++ | EfficientAD | MSFlow | SimpleNet | DSR | **SuperAD (DINOv2)** | **RoBiS (DINOv2-R)** |
|--------|-----------|-----|------|-------------|--------|-----------|-----|---------------------|---------------------|
| Can | 0.3/0.1 | 0.1/0.1 | 0.1/0.1 | 0.8/0.1 | 5.0/0.1 | 0.6/0.1 | 0.4/0.1 | 17.3/1.9 | 1.86/0.84 |
| Fabric | 11.5/9.8 | 2.6/2.2 | 2.9/2.3 | 7.6/1.0 | 22.0/4.1 | 21.6/10.2 | 7.9/5.0 | **77.4/65.3** | **87.46/73.37** |
| Fruit Jelly | 8.7/8.2 | 22.5/22.7 | 26.9/26.7 | 20.8/18.2 | **47.6/38.1** | 25.1/23.0 | 17.9/17.2 | 41.3/40.9 | 53.63/52.62 |
| Rice | 3.8/4.2 | 7.0/3.9 | 9.5/2.9 | 15.0/0.5 | 19.1/1.8 | 11.6/1.0 | 1.5/1.4 | **60.9/61.2** | 63.86/63.23 |
| Sheet Metal | 1.8/1.1 | 41.3/39.2 | 40.9/37.7 | 9.3/3.8 | 13.0/7.6 | 14.6/2.8 | 13.9/14.4 | 59.5/59.7 | **70.98/70.92** |
| Vial | 2.3/2.2 | 28.0/28.3 | 28.2/22.8 | 30.5/26.5 | 23.3/6.2 | 31.9/17.5 | 28.2/27.9 | 42.8/40.8 | **48.73/48.83** |
| Wallplugs | 0.0/0.0 | 1.9/0.8 | 1.3/0.9 | 4.4/0.3 | 0.1/0.2 | 1.0/0.3 | 0.4/0.4 | **13.7/6.7** | 14.38/3.40 |
| Walnuts | 1.2/1.3 | 41.2/36.7 | 44.1/40.5 | 34.6/13.3 | 44.5/14.3 | 35.2/14.3 | 17.0/9.6 | **69.1/69.1** | 67.13/58.94 |
| **Mean** | 3.7/3.4 | 18.1/16.7 | 19.2/16.7 | 15.4/8.0 | 21.8/9.0 | 17.7/8.7 | 10.9/9.5 | **47.8/43.2** | **51.00/46.52** |

#### Table B: RoBiS 추가 메트릭 — TESTpriv / TESTpriv,mix

| Object | AucPro_0.05 (priv) | ClassF1 (priv) | SegF1 (priv) | AucPro_0.05 (mix) | ClassF1 (mix) | SegF1 (mix) |
|--------|---------------------|----------------|-------------|---------------------|----------------|-------------|
| Can | 30.28 | 60.93 | 1.86 | 20.03 | 65.04 | 0.84 |
| Fabric | 79.45 | 83.79 | 87.46 | 79.27 | 83.80 | 73.37 |
| Fruit Jelly | 74.46 | 87.35 | 53.63 | 74.11 | 87.55 | 52.62 |
| Rice | 62.27 | 72.00 | 63.86 | 63.89 | 73.45 | 63.23 |
| Sheet Metal | 75.51 | 87.68 | 70.98 | 73.54 | 86.69 | 70.92 |
| Vial | 76.81 | 84.61 | 48.73 | 69.59 | 85.77 | 48.83 |
| Wallplugs | 62.20 | 75.20 | 14.38 | 24.77 | 72.66 | 3.40 |
| Walnuts | 77.05 | 85.42 | 67.13 | 72.00 | 83.95 | 58.94 |
| **Mean** | **67.25** | **79.62** | **51.00** | **59.65** | **79.86** | **46.52** |

#### Table C: SuperAD (DINOv2) — TESTpub

| Object | AU-ROC_0.05 | F1 Score |
|--------|-------------|----------|
| Can | 58.61 | 0.18 |
| Fabric | 68.29 | 28.22 |
| Fruit Jelly | 80.93 | 48.27 |
| Rice | 92.92 | 68.44 |
| Vial | 69.04 | 35.88 |
| Wallplugs | 77.90 | 19.19 |
| Walnuts | 89.23 | 75.05 |
| Sheet Metal | 76.80 | 40.13 |
| **Mean** | **76.71** | **39.42** |

SuperAD TESTpriv 추가: AucPro_0.05 = **60.51%**, TESTpriv,mix = **58.37%**, ClassF1 = 70.2%/74.4%

### 1.8 로버스트니스 분석 — 핵심 발견

#### 조명 변화에 따른 성능 하락 (TESTpriv → TESTpriv,mix)

**기존 방법 (ResNet/WRN backbone)**:
| Method | SegF1 priv→mix | 하락률 |
|--------|---------------|--------|
| PatchCore | 3.7→3.4 | -8.1% |
| RD | 18.1→16.7 | -7.7% |
| RD++ | 19.2→16.7 | -13.0% |
| EfficientAD | 15.4→8.0 | **-48.1%** |
| MSFlow | 21.8→9.0 | **-58.7%** (AU-PRO에서 51.1% 하락 언급) |
| SimpleNet | 17.7→8.7 | **-50.8%** |
| DSR | 10.9→9.5 | -12.8% |

**FM-based 방법**:
| Method | SegF1 priv→mix | 하락률 | AucPro_0.05 priv→mix | 하락률 |
|--------|---------------|--------|---------------------|--------|
| SuperAD (DINOv2) | 47.8→43.2 | **-9.6%** | 60.51→58.37 | **-3.5%** |
| RoBiS (DINOv2-R+INP-Former) | 51.00→46.52 | **-8.8%** | 67.25→59.65 | **-11.3%** |

#### 핵심 패턴:

1. **가장 취약한 기존 방법**: EfficientAD(-48%), MSFlow(-59%), SimpleNet(-51%) — 학습 시 특정 조명에 과적합
2. **가장 robust한 기존 방법**: PatchCore(-8%), DSR(-13%) — PatchCore는 feature memory bank 기반이라 조명 변화에 덜 민감하나, 절대 성능이 매우 낮음
3. **DINOv2 기반 방법이 절대적 우위**: SuperAD와 RoBiS 모두 기존 방법 대비 2-3배 높은 절대 성능 + 더 작은 상대 하락
4. **카테고리별 편차 극심**:
   - Can: 모든 방법에서 거의 0% SegF1 → 현재 해결 불가 수준
   - Wallplugs: FM 기반에서도 14→3% (RoBiS), 14→7% (SuperAD) → 심한 하락
   - Rice, Sheet Metal: DINOv2 방법에서 조명 변화에 거의 영향 없음 (61→61%, 71→71%)
   - Fabric: DINOv2에서 가장 높은 절대 성능 (87%) + 상대적으로 moderate 하락

#### 왜 특정 방법이 더 robust한가?

1. **DINOv2의 자기지도 학습 특성**: 조명 augmentation 기반 학습으로 내재적 조명 불변성 획득
2. **Training-free 접근 (SuperAD)**: 학습 과정이 없으므로 training set 조명에 과적합 불가
3. **Memory bank 방식의 이점**: 정상 feature 분포 자체에 대한 NN 매칭은 feature 품질에만 의존
4. **학습 기반 방법의 취약점**:
   - EfficientAD의 teacher-student 구조는 training 조명의 reconstruction pattern에 과적합
   - MSFlow의 normalizing flow는 정상 분포를 매우 tight하게 학습 → 조명 변화가 OOD처럼 보임
   - SimpleNet의 경량 discriminator는 학습 조명에서의 결정 경계에 과적합

### 1.9 다운로드 및 사용

```
# 데이터셋 다운로드
# mvtec.com/company/research/datasets/mvtec-ad-2 에서 폼 제출 후 다운로드 링크 수신

# 코드 유틸리티 (PyTorch DataLoader + 평가 스크립트)
# MVTecAD2_public_code_utils.tar.gz

# 평가
# TESTpub: 로컬 평가 가능 (GT 공개)
# TESTpriv, TESTpriv,mix: benchmark.mvtec.com 서버 제출 필수 (GT 비공개)
```

---

## 2. RobustAD (Amazon Science, CVPR 2025 Workshop 추정)

### 2.1 기본 정보
- **제작**: Amazon Science (Latha Pemula, Dongqing Zhang, Onkar Dabeer)
  - 참고: Pemula & Dabeer는 PatchCore와 VisA 논문의 공저자
- **데이터셋 URL**: huggingface.co/datasets/AmazonScience/RobustAD
- **논문**: "COMING SOON" (2026-03-23 기준 미출판)
- **라이선스**: CC-BY-4.0 (상업적 사용 가능 — MVTec AD 2보다 개방적)
- **총 크기**: ~16.27 GB
- **총 이미지**: 2,141+ (viewer 기준, 실제는 더 많을 수 있음)

### 2.2 데이터셋 구성

#### 3개 하위 데이터셋

| 하위 데이터셋 | 산업 | 결함 유형 | 태스크 | Test 도메인 수 |
|-------------|------|---------|--------|-------------|
| **PCB** | 전자/반도체 | Scratches, Soldering melts, Missing parts | Localization + Classification | test0-test5 (6개) |
| **MetalParts** | 자동차 | Chipping, Dents, Porosity | Localization + Classification | test0-test6 (7개) |
| **PiledBags** | 제약 포장 | Count-based anomaly | Classification only | test0-test5 (6개) |

- **해상도**: 2050~3630 pixels (high-resolution)
- **Train split**: 741 images, **Test split**: 1,400 images
- **포맷**: ImageFolder (HuggingFace datasets 호환)
- Label: binary (0=normal, 1=anomaly), Mask: pixel-level (PCB, MetalParts만)

### 2.3 도메인 시프트 구조

```
Train (source domain) → test0, test1, ..., test5/6 (target domains)
```

각 test 도메인은 source domain과 다른 조건에서 촬영/수집된 데이터.
구체적인 도메인 시프트 유형 (조명, 카메라, 배경 등)은 논문 미출판으로 미확인.

**참고**: HuggingFace 데이터셋 카드에서 "9 anomaly types"과 "8 test domains"에 대한 구체적 정보는 아직 공개되지 않음. 현재 확인 가능한 것:
- PCB 6개 + MetalParts 7개 + PiledBags 6개 = 총 19개 test 도메인
- 실제 anomaly types: PCB 3종 + MetalParts 3종 + PiledBags 1종 = 최소 7종 확인

### 2.4 Average Relative Drop (ARD) 메트릭

**논문 미출판으로 정확한 수식 미확인**. 일반적인 ARD 정의:

```
ARD = (1/N) * Σ_i [(M_source - M_target_i) / M_source] * 100%
```

여기서 M_source는 source domain (또는 in-distribution test) 성능, M_target_i는 i번째 target domain 성능.

**의미**: 절대 성능이 아닌 상대적 하락을 측정 → 높은 기본 성능 + 큰 하락 vs 낮은 기본 성능 + 작은 하락을 구분 가능

### 2.5 사용 방법

```python
from datasets import load_dataset, Image

# PCB (Localization + Classification)
pcb_dataset = load_dataset("imagefolder", data_files={
    "train": 'PCB/pcb_data_dir_train/*',
    "test0": 'PCB/pcb_data_dir_test0/*',
    "test1": 'PCB/pcb_data_dir_test1/*',
    "test2": 'PCB/pcb_data_dir_test2/*',
    "test3": 'PCB/pcb_data_dir_test3/*',
    "test4": 'PCB/pcb_data_dir_test4/*',
    "test5": 'PCB/pcb_data_dir_test5/*'
}).cast_column("mask", Image(decode=True))

# MetalParts (Localization + Classification)
metal_parts_dataset = load_dataset("imagefolder", data_files={
    "train": 'MetalParts/metal_parts_data_dir_train/*',
    "test0": 'MetalParts/metal_parts_data_dir_test0/*',
    # ... test1-test6
}).cast_column("mask", Image(decode=True))

# PiledBags (Classification only, no mask)
piled_bags_dataset = load_dataset("imagefolder", data_files={
    "train": 'PiledBags/piled_bags_data_dir_train/*',
    # ... test0-test5
})
```

### 2.6 MVTec AD 2 vs RobustAD 비교

| 차원 | MVTec AD 2 | RobustAD |
|------|-----------|----------|
| **제작사** | MVTec (독일, academic) | Amazon Science (산업) |
| **규모** | 8,004+ images, 8 categories | ~2,141+ images, 3 categories |
| **라이선스** | CC BY-NC-SA 4.0 | CC-BY-4.0 |
| **도메인 시프트 유형** | 조명 변화 (under/over-exposure, additional lights) | 다중 도메인 (구체적 미공개) |
| **시프트 평가** | TESTpriv vs TESTpriv,mix | Source domain vs 6-7 target domains |
| **메트릭** | AU-PRO(0.05), SegF1, ClassF1 | ARD (+ 기본 AD 메트릭) |
| **난이도** | 매우 높음 (SOTA < 60% AU-PRO) | 미확인 (논문 미출판) |
| **GT 접근성** | TESTpub만 공개, TESTpriv 서버 제출 | 전체 공개 |
| **상업 사용** | 불가 | 가능 |
| **논문 상태** | 출판 완료 | 미출판 |
| **핵심 도전** | 투명/반사 객체, 극소 결함, 다양한 조명 | 실제 산업 도메인 시프트 |

**연구 활용 관점**:
- MVTec AD 2는 "조명 조건 하 robustness" 측정에 최적 — 통제된 실험 가능
- RobustAD는 "실제 산업 도메인 시프트" 측정에 적합 — 더 다양한 shift type
- 두 벤치마크를 함께 사용하면 robustness claim의 일반성(generality)을 강화할 수 있음

---

## 3. FM-based 방법의 기존 벤치마크 결과 정리

### 3.1 DINOv2-based 방법 on MVTec AD 2

| 방법 | Backbone | SegF1 (priv/mix) | AucPro_0.05 (priv/mix) | 특징 |
|-----|---------|------------------|----------------------|------|
| **SuperAD** | DINOv2-ViT-L-14 | 47.8/43.2 | 60.51/58.37 | Training-free, 16-shot memory bank, 4-layer features |
| **RoBiS** | DINOv2-R + ViT-B-14 (INP-Former) | 51.00/46.52 | 67.25/59.65 | Swin-Crop + noise/lighting augmentation + MEBin + SAM |

**핵심 관찰**:
- DINOv2가 기존 ResNet/WRN 기반 방법 대비 2-3x 높은 성능
- Training-free (SuperAD)가 training-based와 비슷하거나 약간 낮은 성능 → DINOv2 feature 품질 자체가 핵심
- RoBiS의 lighting augmentation이 추가 robustness 제공 (AucPro_0.05: 67.25 vs 60.51)
- 그러나 RoBiS는 mix에서 AucPro_0.05 11.3% 하락 vs SuperAD의 3.5% → augmentation이 오히려 domain-specific bias 도입 가능성

### 3.2 CLIP-based 방법 on MVTec AD 2

직접적인 결과 미발견. 그러나:
- APRIL-GAN (CLIP 기반)은 MVTec AD 2 논문에서 벤치마크되지 않음
- AnomalyVFM (arXiv:2601.20524) 연구에 따르면 DINOv2가 CLIP보다 AD에서 underperform하는 것은 adaptation 부족 때문이며, 적절한 adaptation 시 DINOv2가 CLIP을 능가

### 3.3 기타 관련 결과

| 방법 | Backbone | MVTec AD 성능 | MVTec AD 2 성능 | 하락 |
|-----|---------|-------------|---------------|------|
| EfficientAD | PDN (경량) | ~99% AUROC | 15.4% SegF1 (58.7% AU-PRO) | 극심 |
| PatchCore | WRN-50-2 | ~99% AUROC | 3.7% SegF1 (53.8% AU-PRO) | 극심 |
| MSFlow | ResNet | ~98% AUROC | 21.8% SegF1 | 극심 + 조명 취약 |

→ **MVTec AD에서의 거의 완벽한 성능은 MVTec AD 2에서 완전히 무너짐**

---

## 4. 평가 프로토콜 설계 제안

### 4.1 메트릭 체계

#### 필수 메트릭
| 메트릭 | 용도 | 비고 |
|--------|------|------|
| **AU-PRO(0.05)** | Pixel-level localization (엄격) | MVTec AD 2 공식 메트릭 |
| **SegF1** | Binary segmentation 성능 | VAND 3.0 챌린지 메트릭 |
| **Image-level AUROC** | Classification 성능 | 표준 비교용 |
| **ClassF1** | Classification F1 | Image-level 성능 |

#### Robustness 전용 메트릭
| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **Absolute Drop (AD)** | M_priv - M_mix | 절대 성능 하락 |
| **Relative Drop (RD)** | (M_priv - M_mix) / M_priv * 100% | 상대 성능 하락 (ARD와 유사) |
| **Robustness Ratio (RR)** | M_mix / M_priv | 1에 가까울수록 robust |
| **Performance-Robustness Product** | M_priv * RR | 절대 성능과 robustness 동시 고려 |

#### 보조 분석 메트릭
| 메트릭 | 용도 |
|--------|------|
| **Category-level variance** | 성능 안정성 측정 (std across categories) |
| **Worst-case category** | 가장 취약한 카테고리 성능 |
| **Size-stratified PRO** | 결함 크기별 검출 성능 |
| **FPR at fixed TPR** | 실전 활용 가능성 |

### 4.2 실험 구조

```
Phase 1: In-Distribution 성능 (TESTpriv)
├── AU-PRO(0.05), SegF1, ClassF1 per category
├── 카테고리별 분석 (어떤 유형에서 강/약)
└── Baseline 대비 비교

Phase 2: Robustness 평가 (TESTpriv,mix)
├── 동일 메트릭 반복
├── Absolute/Relative Drop 계산
├── Category-level robustness 분석
└── 어떤 카테고리에서 가장 robust/fragile한지

Phase 3: Cross-benchmark 검증 (RobustAD)
├── 동일한 method를 RobustAD에서 평가
├── Domain별 성능 curve
├── MVTec AD 2 결과와의 상관관계 분석
└── Robustness claim의 일반화 검증

Phase 4: 분석 실험
├── Feature visualization under lighting shift
├── Layer-wise robustness 분석
├── Ablation: backbone, adaptation, memory bank size
└── Feature disentanglement 효과 측정
```

### 4.3 통계 검정

**Robustness claim을 위한 통계적 근거**:

1. **Paired comparison**: TESTpriv vs TESTpriv,mix는 동일 카테고리의 paired data → Wilcoxon signed-rank test 또는 paired t-test
2. **Effect size**: Cohen's d 또는 Cliff's delta — 단순 p-value 넘어 효과 크기 보고
3. **Confidence interval**: Bootstrap CI for mean AU-PRO drop
4. **Multiple comparison correction**: 8 categories x multiple methods → Bonferroni 또는 FDR correction
5. **Seed robustness**: 학습 기반 방법은 최소 3 seeds, mean +/- std 보고

**참고**: MVTec AD 2는 서버 제출 기반이므로 제출 횟수에 제한이 있을 수 있음 → seed 반복 실험은 TESTpub에서 수행하고 TESTpriv/mix는 최종 모델로 1회 제출이 현실적

### 4.4 우리 연구에 특화된 평가 포인트

Feature disentanglement/robustness 연구라면:

1. **Feature space 분석**:
   - Lighting shift 전후 feature 분포 변화 시각화 (t-SNE, UMAP)
   - Normal feature의 shift magnitude vs anomaly score의 shift magnitude 비교
   - 어떤 feature dimension이 lighting에 민감한지 분석

2. **Disentanglement 효과 측정**:
   - Nuisance-removed feature로 AD → robustness 개선 확인
   - Anomaly-relevant feature 보존 확인 (in-distribution 성능 유지)
   - Category별 disentanglement 효과 차이 분석

3. **Comparison axes**:
   - vs DINOv2 naive (SuperAD 재현) → disentanglement의 marginal effect
   - vs Augmentation-based robustness (RoBiS 방식) → 우리 방법의 차별성
   - vs TTA/domain adaptation baseline → 문제 접근 방식의 차이

---

## 5. 핵심 Takeaways for Our Research

1. **MVTec AD 2가 primary benchmark**: 가장 challenging하고 잘 정의된 robustness 평가. TESTpriv vs TESTpriv,mix 비교가 우리 연구의 핵심 evidence.

2. **DINOv2가 이미 strong baseline**: SuperAD의 training-free DINOv2가 기존 방법 대비 압도적. 우리의 contribution은 DINOv2를 "더 잘 쓰는" 것이 아니라, "왜 robust한지/어디서 실패하는지"를 분석하고, 체계적으로 개선하는 것이어야 함.

3. **Can, Wallplugs가 핵심 분석 대상**: 모든 방법이 실패하는 카테고리 → 실패 원인 분석이 논문의 핵심 contribution이 될 수 있음.

4. **조명 robustness 외에도**: MVTec AD 2의 투명/반사 객체, 극소 결함 등은 조명과 독립적인 도전 → robustness와 difficulty를 분리해서 분석해야 함.

5. **RobustAD는 보조 벤치마크**: 논문 미출판, 상세 미공개. 그러나 MVTec AD 2와 다른 shift type을 제공하므로 generalization claim에 유용. GT가 전체 공개되어 반복 실험 용이.

6. **통계적 rigor 필수**: 8개 카테고리 × 2개 test set × 다수 메트릭이므로 multiple testing 이슈 존재. Effect size + CI 보고 필수.

---

## 관련 노트
- [2026-03-23_failure_survey.md](2026-03-23_failure_survey.md) — FM-AD 실패 모드 분석
- [2026-03-23_mechanism_analysis.md](2026-03-23_mechanism_analysis.md) — FM feature robustness 메커니즘
- [2026-03-23_solution_survey.md](2026-03-23_solution_survey.md) — 기존 해결 접근
- [2026-03-23_ICLR_positioning.md](2026-03-23_ICLR_positioning.md) — 논문 포지셔닝

## 참고 논문
- MVTec AD 2: arXiv:2503.21622 (Heckler-Kram et al., 2025)
- SuperAD: arXiv:2505.19750 (Zhang et al., 2025)
- RoBiS: arXiv:2505.21152 (Li et al., 2025)
- MVTec AD (original): Bergmann et al., CVPR 2019
- PatchCore: arXiv:2106.08265 (Roth et al., 2022)
- VisA: arXiv:2207.14315 (Zou et al., 2022)
- AnomalyVFM: arXiv:2601.20524
- Robust Distribution Alignment: arXiv:2503.14910
